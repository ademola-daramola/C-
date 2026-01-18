#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>
#include <limits>

using namespace std;

// --- CONFIGURATION CONSTANTS ---
const int POPULATION_SIZE = 50;
const int GENERATIONS = 100;
const double CROSSOVER_PROB = 0.80;
const double MUTATION_PROB = 0.25;
const int BITS_PER_VAR = 10; // Interpreting "10 byte length" as 10-bit precision for feasibility
const int NUM_VARS = 4;
const int CHROMOSOME_LENGTH = BITS_PER_VAR * NUM_VARS;

// --- DOMAIN BOUNDS ---
const double X1_MIN = 1.0, X1_MAX = 3.0;
const double X2_MIN = 0.0, X2_MAX = 5.0;
const double X3_MIN = 1.0, X3_MAX = 2.0; 
const double X4_MIN = 0.0, X4_MAX = 4.0;

// --- RANDOM NUMBER GENERATOR ---
random_device rd;
mt19937 gen(rd());

// --- INDIVIDUAL STRUCTURE ---
struct Individual {
    string chromosome;
    double x[4]; // Decoded values x1, x2, x3, x4
    double fitness;

    Individual() {
        chromosome = "";
        fitness = -numeric_limits<double>::infinity();
    }
};

// --- HELPER FUNCTIONS ---

// 1. Generate random binary string
string randomChromosome(int length) {
    string s = "";
    uniform_int_distribution<> dis(0, 1);
    for (int i = 0; i < length; ++i) {
        s += to_string(dis(gen));
    }
    return s;
}

// 2. Decode Binary to Real Value
// Formula: Val = Min + (Decimal / (2^Bits - 1)) * (Max - Min)
double decode(string binSegment, double minVal, double maxVal) {
    unsigned long decimal = stoul(binSegment, nullptr, 2);
    double maxDecimal = pow(2, BITS_PER_VAR) - 1;
    return minVal + (double(decimal) / maxDecimal) * (maxVal - minVal);
}

// 3. Update Individual's Phenotype (x1...x4) and Fitness
void evaluate(Individual& ind) {
    // Slice chromosome into 4 parts
    string s1 = ind.chromosome.substr(0 * BITS_PER_VAR, BITS_PER_VAR);
    string s2 = ind.chromosome.substr(1 * BITS_PER_VAR, BITS_PER_VAR);
    string s3 = ind.chromosome.substr(2 * BITS_PER_VAR, BITS_PER_VAR);
    string s4 = ind.chromosome.substr(3 * BITS_PER_VAR, BITS_PER_VAR);

    ind.x[0] = decode(s1, X1_MIN, X1_MAX);
    ind.x[1] = decode(s2, X2_MIN, X2_MAX);
    ind.x[2] = decode(s3, X3_MIN, X3_MAX);
    ind.x[3] = decode(s4, X4_MIN, X4_MAX);

    double x1 = ind.x[0];
    double x2 = ind.x[1];
    double x3 = ind.x[2];
    double x4 = ind.x[3];

    // Function from Image 1
    ind.fitness = (2 * x1 * x2 * x3 * x4) - (4 * x1 * x2 * x3) - (2 * x2 * x3 * x4)
        - (x1 * x2) - (x3 * x4)
        + (x1 * x1) + (x2 * x2) + (x3 * x3) + (x4 * x4)
        - (2 * x1) - (4 * x2) + (4 * x3) - (2 * x4);
}

// 4. Roulette Wheel Selection
// Note: Handles negative fitness by shifting values to be positive
int rouletteSelect(const vector<Individual>& pop) {
    // Find min fitness to normalize
    double minFit = pop[0].fitness;
    for (const auto& ind : pop) {
        if (ind.fitness < minFit) minFit = ind.fitness;
    }

    // Create adjusted fitness weights (must be > 0)
    vector<double> weights;
    double sumWeights = 0.0;
    for (const auto& ind : pop) {
        // Shift so worst fitness is slightly above 0
        double w = ind.fitness - minFit + 0.01;
        weights.push_back(w);
        sumWeights += w;
    }

    uniform_real_distribution<> dis(0, sumWeights);
    double r = dis(gen);
    double cumulative = 0.0;

    for (size_t i = 0; i < pop.size(); ++i) {
        cumulative += weights[i];
        if (cumulative >= r) return i;
    }
    return pop.size() - 1;
}

// 5. Crossover (Single Point)
pair<Individual, Individual> crossover(Individual p1, Individual p2) {
    uniform_real_distribution<> dis(0.0, 1.0);
    if (dis(gen) > CROSSOVER_PROB) {
        return { p1, p2 }; // No crossover
    }

    uniform_int_distribution<> pointDis(1, CHROMOSOME_LENGTH - 1);
    int point = pointDis(gen);

    Individual c1, c2;
    c1.chromosome = p1.chromosome.substr(0, point) + p2.chromosome.substr(point);
    c2.chromosome = p2.chromosome.substr(0, point) + p1.chromosome.substr(point);

    return { c1, c2 };
}

// 6. Mutation (Bit Flip)
void mutate(Individual& ind) {
    uniform_real_distribution<> dis(0.0, 1.0);
    // Determine if this individual mutates
    if (dis(gen) < MUTATION_PROB) {
        // Flip one random bit
        uniform_int_distribution<> bitDis(0, CHROMOSOME_LENGTH - 1);
        int bit = bitDis(gen);
        ind.chromosome[bit] = (ind.chromosome[bit] == '0') ? '1' : '0';
    }
}

int main() {
    // A. INITIALIZATION
    vector<Individual> population(POPULATION_SIZE);
    Individual globalBest;

    // Create initial random population
    for (int i = 0; i < POPULATION_SIZE; ++i) {
        population[i].chromosome = randomChromosome(CHROMOSOME_LENGTH);
        evaluate(population[i]);
    }

    // Set initial global best
    globalBest = population[0];

    // Print Header
    cout << "Gen\tBestFit\tGlobalBestFit\tx1\tx2\tx3\tx4" << endl;
    cout << "-------------------------------------------------------------" << endl;

    // B. GENERATION LOOP
    for (int gen = 0; gen < GENERATIONS; ++gen) {

        // 1. Find best in current generation
        Individual currentBest = population[0];
        for (const auto& ind : population) {
            if (ind.fitness > currentBest.fitness) {
                currentBest = ind;
            }
        }

        // 2. Update Global Best (Best ever seen)
        if (currentBest.fitness > globalBest.fitness) {
            globalBest = currentBest;
        }

        // 3. Report Data (For plotting later)
        cout << gen + 1 << "\t"
            << fixed << setprecision(4) << currentBest.fitness << "\t"
            << globalBest.fitness << "\t"
            << currentBest.x[0] << "\t" << currentBest.x[1] << "\t"
            << currentBest.x[2] << "\t" << currentBest.x[3] << endl;

        // 4. Create Next Generation
        vector<Individual> nextGen;

        while (nextGen.size() < POPULATION_SIZE) {
            // Selection
            int p1_idx = rouletteSelect(population);
            int p2_idx = rouletteSelect(population);

            // Crossover
            pair<Individual, Individual> children = crossover(population[p1_idx], population[p2_idx]);

            // Mutation
            mutate(children.first);
            mutate(children.second);

            // Evaluate
            evaluate(children.first);
            evaluate(children.second);

            // Fill new population
            nextGen.push_back(children.first);
            if (nextGen.size() < POPULATION_SIZE) nextGen.push_back(children.second);
        }

        population = nextGen;
    }

    // C. FINAL REPORT
    cout << "\n=============================================" << endl;
    cout << "FINAL RESULTS" << endl;
    cout << "=============================================" << endl;
    cout << "Optimum Value Found (Fitness): " << globalBest.fitness << endl;
    cout << "Variable Values:" << endl;
    cout << "x1 = " << globalBest.x[0] << endl;
    cout << "x2 = " << globalBest.x[1] << endl;
    cout << "x3 = " << globalBest.x[2] << endl;
    cout << "x4 = " << globalBest.x[3] << endl;
    cout << "Binary Chromosome: " << globalBest.chromosome << endl;

    return 0;
}