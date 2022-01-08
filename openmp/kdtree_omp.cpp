#include <algorithm>
#include <iostream>
#include <functional>
#include <chrono>
#include <math.h>

#include "Utility.hpp"
#include <immintrin.h>
#include <omp.h>
#include <assert.h>

#define DEBUG 1

// Here we introduce a new struct data structure, to avoid returining any value in nearest function which makes the parallelism easier
struct Nearest_node
{
    Nearest_node(Node* bg, float bgd): best_global(bg), best_global_dist(bgd){}
    Node* best_global;
    float best_global_dist;

};
/***************************************************************************************/
// =======================================================================
float Point::distance_squared(Point &a, Point &b)
{
    if(a.dimension != b.dimension)
    {
        std::cout << "Dimensions do not match!" << std::endl;
        exit(1);
    }
    float dist = 0;

    for(int i = 0; i < a.dimension; ++i)
    {
        float tmp = a.coordinates[i] - b.coordinates[i];
        dist += tmp * tmp;
    }
    return dist;
}
/***************************************************************************************/

/***************************************************************************************/
Node* build_tree_rec(Point** point_list, int num_points, int depth)
{
    if (num_points <= 0)
    {
        return nullptr;
    }

    if (num_points == 1)
    {
        return new Node(point_list[0], nullptr, nullptr);
    }

    int dim = point_list[0]->dimension;

    // sort list of points based on axis
    int axis = depth % dim; // to know w.r.t. which coordinate we should sort the points
    using std::placeholders::_1;
    using std::placeholders::_2;

    // We have used lambda function for sorting instead of calling compare (saves us some time)
    std::sort(point_list, point_list + (num_points - 1),
              std::bind([](Point *a, Point *b, int axis){return a->coordinates[axis] < b->coordinates[axis];}, _1,_2, axis));



    // select median
    Point** median = point_list + (num_points / 2); // pointer to the median point w.r.t. to coordinate axis
    Point** left_points = point_list; // pointer to the start of the left points
    Point** right_points = median + 1; // pointer to the start of the right points

    int num_points_left = num_points / 2;
    int num_points_right = num_points - (num_points / 2) - 1;

    //  Let;s build the tree in parallel by creating a task for each branch
    Node* left_node;
    Node* right_node;
    // left subtree : create an omp task
    #pragma omp task shared(left_node) if(depth < 12)
        left_node = build_tree_rec(left_points, num_points_left, depth + 1);

    // right subtree : create an omp task
    #pragma omp task shared(right_node) if(depth < 12)
        right_node = build_tree_rec(right_points, num_points_right, depth + 1);

    // wait for the tasks to be complete before returning
    #pragma omp taskwait 

    // return median node
    return new Node(*median, left_node, right_node);
}

Node* build_tree(Point** point_list, int num_nodes)
{
    Node* node;
    #pragma omp parallel
    {
        #pragma omp single
        node = build_tree_rec(point_list, num_nodes, 0);
    };
    return node;
}
/***************************************************************************************/

/***************************************************************************************/

void nearest(Node* root, Point* query, int depth, Nearest_node* The_best)
{
    // leaf node
    if (root == nullptr)
    {
        return;
    }

    // int dim = query->dimension;
    // int axis = depth % dim; // Coordinate w.r.t. which we comapre at the current depth
    int axis = depth % (query->dimension);

    float d_euclidian = root->point->distance_squared(*query); // Euclidean distance between current point and the query point

    float d_axis = query->coordinates[axis] - root->point->coordinates[axis]; // axis distance between current point and the query point along the axis
    float d_axis_squared = d_axis * d_axis; // square the axis distance for comparison

    //  Update the best_local and best_dist_local if current point is closer to the query
    if (d_euclidian < The_best->best_global_dist)
    {
        // update the global information (which allows us to avoid returning any value)
        The_best->best_global = root;
        The_best->best_global_dist = d_euclidian;
    }

    // decide which branch to visit based on where the query point is located w.r.t. the axis
    Node* visit_branch;
    Node* other_branch;

    if(d_axis < 0)
    {
        // query point is smaller than root node in axis dimension, i.e. go left
        visit_branch = root->left;
        other_branch = root->right;
    }
    else
    {
        // query point is larger than root node in axis dimension, i.e. go right
        visit_branch = root->right;
        other_branch = root->left;
    }

    if (depth<1)
    {
        // Here let's check both sides anyway in parallel.
        #pragma omp parallel sections shared(The_best)
        {
            #pragma omp section
            {
                nearest(visit_branch, query, depth + 1, The_best);
            }
            #pragma omp section
            {
                nearest(other_branch, query, depth + 1,  The_best);
            }
        }
    }
    else
    {
        // After we exceed a specific depth, let's continue in a sequential way, to avoid the overhead created by the parallelism
        nearest(visit_branch, query, depth + 1, The_best);

        if (d_axis_squared < The_best->best_global_dist) // check the other side if the ball intesects with the hyperplane.
        {
            nearest(other_branch, query, depth + 1, The_best);
        }

    }
}

Node* nearest_neighbor(Node* root, Point* query)
{
    float best_dist = root->point->distance_squared(*query);
    Nearest_node* The_best = new Nearest_node(root, best_dist);
    nearest(root, query, 0, The_best); // Here we pass a pointer to the new struct data structure
    return The_best->best_global; // it still returns a pointer to the nearest node
}


/***************************************************************************************/
int main(int argc, char **argv)
{
    omp_set_num_threads(12); // Use only 12 threads as we want to get the required speed up on a 6 core machine
    int seed = 0;
    int dim = 0;
    int num_points = 0;
    int num_queries = 10;

#if DEBUG
    // for measuring your local runtime
    auto tick = std::chrono::high_resolution_clock::now();
    Utility::specify_problem(argc, argv, &seed, &dim, &num_points);
#else
    Utility::specify_problem(&seed, &dim, &num_points);
#endif

    // last points are query
    float* x = Utility::generate_problem(seed, dim, num_points + num_queries);
    Point** points = (Point**)calloc(num_points, sizeof(Point*));

    for(int n = 0; n < num_points; ++n)
    {
        points[n] = new Point(dim, n + 1, x + n * dim);
    }

    // build tree
    Node* tree = build_tree(points, num_points);

    // for each query, find nearest neighbor
    Node* res[num_queries]; // new data structure to parallelize the queries
    Point queries[num_queries]; // new data structure to parallelize the queries
    
    #pragma omp parallel for
        for(int q = 0; q < num_queries; ++q)
        {
            float* x_query = x + (num_points + q) * dim;
            Point query(dim, num_points + q, x_query);
            queries[q] = query;

            res[q] = nearest_neighbor(tree, &query);

        }

    for(int q = 0; q < num_queries; ++q)
    {
        // output min-distance (i.e. to query point)
        float min_distance = queries[q].distance(*res[q]->point);
        Utility::print_result_line(queries[q].ID, min_distance);

#if DEBUG
        // in case you want to have further debug information about
        // the query point and the nearest neighbor
        std::cout << "Query: " << queries[q] << std::endl;
        std::cout << "NN: " << *res[q]->point << std::endl << std::endl;
#endif
    }

#if DEBUG
    // for measuring your local runtime
    auto tock = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = tock - tick;
    std::cout << "elapsed time " << elapsed_time.count() << " second" << std::endl;
#endif

    std::cout << "DONE" << std::endl;

    // clean-up
    Utility::free_tree(tree);

    for(int n = 0; n < num_points; ++n)
    {
        delete points[n];
    }

    free(points);
    free(x);

    (void)argc;
    (void)argv;
    return 0;
}
/***************************************************************************************/