#include <algorithm>
#include <iostream>
#include <functional>
#include <chrono>
#include <random>
#include <math.h>
#include <mpi.h>
#include <cfloat>
#include <omp.h>

#include "Utility.hpp"

#define DEBUG 0

// Here we introduce a new struct data structure, to avoid returining any value in nearest function which makes the parallelism easier
struct Nearest_node
{
    Nearest_node(Node* bg, float bgd): best_global(bg), best_global_dist(bgd){}
    Node* best_global;
    float best_global_dist;

};

/***************************************************************************************/
float Point::distance_squared(Point &a, Point &b){
    if(a.dimension != b.dimension){
        std::cout << "Dimensions do not match!" << std::endl;
        exit(1);
    }
    float dist = 0;
    for(int i = 0; i < a.dimension; ++i){
        float tmp = a.coordinates[i] - b.coordinates[i];
        dist += tmp * tmp;
    }
    return dist;
}
/***************************************************************************************/


/***************************************************************************************/
Node* build_tree_rec(Point** point_list, int num_points, int depth){
    if (num_points <= 0){
        return nullptr;
    }

    if (num_points == 1){
        return new Node(point_list[0], nullptr, nullptr);
    }

    int dim = point_list[0]->dimension;

    // sort list of points based on axis
    int axis = depth % dim;
    using std::placeholders::_1;
    using std::placeholders::_2;

    std::sort(
            point_list, point_list + (num_points - 1),
            std::bind(Point::compare, _1, _2, axis));

    // select median
    Point** median = point_list + (num_points / 2);
    Point** left_points = point_list;
    Point** right_points = median + 1;

    int num_points_left = num_points / 2;
    int num_points_right = num_points - (num_points / 2) - 1;

    Node* left_node;
    Node* right_node;

    // left subtree
    #pragma omp task shared(left_node) if(depth < 5)
        left_node = build_tree_rec(left_points, num_points_left, depth + 1);

    // right subtree
    #pragma omp task shared(right_node) if(depth < 5)
        right_node = build_tree_rec(right_points, num_points_right, depth + 1);

    #pragma omp taskwait

    // return median node
    return new Node(*median, left_node, right_node);
}

Node* build_tree(Point** point_list, int num_nodes){
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

    #pragma omp task shared(The_best) if(depth < 3)
    {
        nearest(visit_branch, query, depth + 1, The_best);
        if (d_axis_squared < The_best->best_global_dist) {
            nearest(other_branch, query, depth + 1,  The_best);
        }
    }

}


Node* nearest_neighbor(Node* root, Point* query)
{
    float best_dist = root->point->distance_squared(*query);
    Nearest_node* The_best = new Nearest_node(root, best_dist);
    #pragma omp parallel
    {
        #pragma omp single
            nearest(root, query, 0, The_best); // Here we pass a pointer to the new struct data structure
    }
    return The_best->best_global; // it still returns a pointer to the nearest node
}


/***************************************************************************************/
int main(int argc, char **argv){
    omp_set_num_threads(12);
    int seed = 0;
    int dim = 0;
    int num_points = 0;
    int num_queries = 10;

    int rank, num_procs;
    int prov;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &prov);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0) { // master
        // specify the problem only in master
        Utility::specify_problem(&seed, &dim, &num_points);
    }

    // broadcast the input parameters (defined in master) to workers
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) // MASTER ROUTINE
    {
        // stores global_min_distance per query. initialized with FLOAT_MAX
        float minDistanceArray[num_queries];
        for (int i=0; i<num_queries; i++) {
            minDistanceArray[i] = FLT_MAX;
        }

        // master expects numTotalMessages from workers
        int numTotalMessages = num_queries * (num_procs - 1);

        for (int q=0; q < numTotalMessages; q++) {
            float currentDistanceResult;
            MPI_Status status;
            MPI_Recv(&currentDistanceResult, 1, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            // worker sends the query number in MPI_TAG
            int senderTag = status.MPI_TAG;

            // update global_min_distance if local_min_distance is less than global
            if (currentDistanceResult < minDistanceArray[senderTag]) {
                minDistanceArray[senderTag] = currentDistanceResult;
            }
        }

        // print the global_min_distance results after all messages received from workers
        for(int i=0; i<num_queries; i++) {
            Utility::print_result_line(num_points + i, minDistanceArray[i]);
        }
        std::cout << "DONE" << std::endl;
    }
    else // WORKER ROUTINE
    {
        float* x = Utility::generate_problem(seed, dim, num_points + num_queries);

        // let each worker deal with only a specific part of the points (from start to end indexes)
        int avgPointsPerWorker = num_points / (num_procs - 1);
        int start = (avgPointsPerWorker) * (rank - 1);
        int end;

        // let last ranked worker process until the last point
        if (rank == num_procs - 1) {
            end = num_points;
        }
        else {
            end = start + avgPointsPerWorker;
        }

        int numPointsOfThisWorker = end - start;
        Point** points = (Point**)calloc(numPointsOfThisWorker, sizeof(Point*));
        for (int n = start; n < end; ++n) {
            points[n - start] = new Point(dim, n + 1, x + n * dim);
        }

        // each worker builds a tree of a specific part of points
        Node* tree = build_tree(points, numPointsOfThisWorker);

        // each worker runs all the queries on their local trees
        #pragma omp parallel for
        for (int q = 0; q < num_queries; ++q) {
            float* x_query = x + (num_points + q) * dim;
            Point query(dim, num_points + q, x_query);

            Node *res = nearest_neighbor(tree, &query);
            float min_distance = query.distance(*res->point);

            // each worker sends the min_distance calculated on their local trees.
            // the message is tagged with q (current query_number)
            MPI_Send(&min_distance, 1, MPI_FLOAT, 0, q, MPI_COMM_WORLD);
        }

        // clean-up
        Utility::free_tree(tree);
        for(int n = 0; n < numPointsOfThisWorker; ++n){
            delete points[n];
        }
        free(points);
        free(x);
    }

    (void)argc;
    (void)argv;
    MPI_Finalize();

    return 0;
}
/***************************************************************************************/