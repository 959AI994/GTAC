#include "database.h"

using namespace std;

const std::set<ll> areaOptFuncs_2Input = {
    0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15
};

const std::set<ll> areaOptCanoFuncs_3Input = {
    0, 1, 6, 7, 17, 23, 25, 27, 30, 85, 102, 129
};

const std::set<ll> areaOptCanoFuncs_4Input = {
    0, 1, 6, 7, 22, 23, 25, 27, 30, 31, 61, 105, 111, 126, 127, 129, 151, 257, 279, 281, 282, 283, 286, 287, 300, 301, 303, 316, 317, 318, 319, 361, 367, 383, 385, 387, 391, 393, 399, 408, 409, 411, 429, 445, 447, 489, 495, 510, 583, 598, 599, 607, 727, 829, 856, 857, 858, 859, 862, 863, 876, 879, 892, 893, 894, 961, 966, 967, 983, 984, 985, 987, 990, 1083, 1443, 1467, 1542, 1632, 1638, 1639, 1647, 1654, 1662, 1686, 1695, 1782, 1785, 1799, 1910, 1912, 1914, 1972, 1973, 1980, 2017, 2019, 2025, 2040, 2167, 2449, 2677, 3025, 4105, 4110, 4111, 4239, 4369, 4382, 4383, 4494, 4495, 5070, 5547, 5654, 5782, 5790, 5911, 6120, 6205, 6270, 6273, 6297, 6425, 6552, 6553, 6630, 6939, 7140, 7710, 7905, 10327, 16683, 21845, 24591, 26214, 32769, 32791, 33055, 33150, 33153, 33367, 37150
};

void saveToCSV(const std::map<std::vector<ll>, double>& table, const std::string& filename) {
    // cout << "begin saveToCSV!" << endl;
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << '\n';
        return;
    }

    for (const auto& [keyVec, value] : table) {
        for (size_t i = 0; i < keyVec.size(); ++i) {
            file << keyVec[i];
            if (i + 1 < keyVec.size()) file << ",";
        }
        file << "," << value << "\n";
    }

    file.close();
    // cout << "finish saveToCSV!" << endl;
}

void saveToBin(const std::map<std::vector<ll>, double>& table, const std::string& filename) {
    // cout << "begin saveToBin!" << endl;
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << '\n';
        return;
    }

    // Write map size
    size_t mapSize = table.size();
    file.write(reinterpret_cast<const char*>(&mapSize), sizeof(mapSize));

    for (const auto& [vec, value] : table) {
        // Write vector size
        size_t vecSize = vec.size();
        file.write(reinterpret_cast<const char*>(&vecSize), sizeof(vecSize));

        // Write vector elements
        file.write(reinterpret_cast<const char*>(vec.data()), vecSize * sizeof(ll));

        // Write double value
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    file.close();
    // cout << "Finish saveToBin!" << endl;
}

// void Database::loadBinMap(const std::string& filename, std::map<std::vector<ll>, double>& table) {
//     std::ifstream file(filename, std::ios::binary);
//     if (!file.is_open()) {
//         std::cerr << "Failed to open file: " << filename << '\n';
//         return;
//     }

//     table.clear();

//     size_t mapSize;
//     file.read(reinterpret_cast<char*>(&mapSize), sizeof(mapSize));

//     for (size_t i = 0; i < mapSize; ++i) {
//         size_t vecSize;
//         file.read(reinterpret_cast<char*>(&vecSize), sizeof(vecSize));

//         std::vector<ll> vec(vecSize);
//         file.read(reinterpret_cast<char*>(vec.data()), vecSize * sizeof(ll));

//         double value;
//         file.read(reinterpret_cast<char*>(&value), sizeof(value));

//         table[vec] = value;
//     }

//     file.close();
//     cout << "finish loading " << filename << endl;
// }

void Database::loadBinMap(const std::string& filename, std::map<std::vector<ll>, double>& table) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << '\n';
        return;
    }

    table.clear();

    size_t mapSize;
    if (!file.read(reinterpret_cast<char*>(&mapSize), sizeof(mapSize))) {
        std::cerr << "File is empty: " << filename << '\n';
        return;
    }

    for (size_t i = 0; i < mapSize; ++i) {
        size_t vecSize;
        if (!file.read(reinterpret_cast<char*>(&vecSize), sizeof(vecSize))) break;

        std::vector<ll> vec(vecSize);
        if (!file.read(reinterpret_cast<char*>(vec.data()), vecSize * sizeof(ll))) break;

        double value;
        if (!file.read(reinterpret_cast<char*>(&value), sizeof(value))) break;

        table[vec] = value;
    }

    file.close();
    cout << "Finished loading " << filename << endl;
}

bool Database::loadBinNPN(const std::string& filepath, int nVars, std::vector<NpnInfo>& targetVec) {
    const size_t numEntries = 1ULL << (1ULL << nVars); // 2^(2^n)
    targetVec.resize(numEntries);

    std::ifstream fin(filepath, std::ios::binary);
    if (!fin) {
        std::cerr << "Failed to open file: " << filepath << "\n";
        return false;
    }

    for (size_t i = 0; i < numEntries; ++i) {
        NpnInfo info;
        fin.read(reinterpret_cast<char*>(&info.canoFunc), sizeof(uint16_t));
        fin.read(reinterpret_cast<char*>(&info.permutation[0]), 4);
        fin.read(reinterpret_cast<char*>(&info.input_inversion), 1);
        uint8_t out_inv = 0;
        fin.read(reinterpret_cast<char*>(&out_inv), 1);
        info.output_inversion = (out_inv != 0);

        if (!fin) {
            std::cerr << "Error reading entry at index " << i << " from file: " << filepath << "\n";
            return false;
        }

        targetVec[i] = info;
    }

    std::cout << "Loaded " << numEntries << " entries from " << filepath << "\n";
    return true;
}

void Database::loadNpnDB() {
    assert(loadBinNPN("database/npn_cano_3bit.bin", 3, npn3));
    assert(loadBinNPN("database/npn_cano_4bit.bin", 4, npn4));
}

// void Database::loadMapDB() {
//     struct MapLoadTask {
//         const char* filename;
//         std::map<std::vector<ll>, double>* target;
//     };

//     std::vector<MapLoadTask> tasks = {
//         {"database/map21.bin", &map21},
//         {"database/map22.bin", &map22},
//         {"database/map23.bin", &map23},
//         {"database/map31.bin", &map31},
//         {"database/map32.bin", &map32},
//         {"database/map33.bin", &map33},
//         {"database/map41.bin", &map41},
//         {"database/map42.bin", &map42},
//         {"database/map43.bin", &map43}
//     };

//     for (const auto& task : tasks) {
//         loadBinMap(task.filename, *task.target);
//     }
// }

void Database::loadSuppDB() {
    struct MapLoadTask {
        const char* filename;
        std::map<std::vector<ll>, double>* target;
    };

    std::vector<MapLoadTask> tasks = {
        {"database_supp/suppMap32.bin", &suppMap32},
        {"database_supp/suppMap33.bin", &suppMap33},
        {"database_supp/suppMap42.bin", &suppMap42},
        {"database_supp/suppMap43.bin", &suppMap43}
    };

    for (const auto& task : tasks) {
        loadBinMap(task.filename, *task.target);
    }
}

void Database::loadMMapDB() {
    struct MapLoadTask {
        const char* filename;
        std::multimap<double, std::vector<ll>>* target;
    };

    std::vector<MapLoadTask> tasks = {
        {"database_area_opt_haveNonCano/areaOptMap21.bin", &areaOptMap21},
        // {"database_area_opt_haveNonCano/areaOptMap22.bin", &areaOptMap22},
        // {"database_area_opt_haveNonCano/areaOptMap23.bin", &areaOptMap23},
        {"database_area_opt_haveNonCano/areaOptMap31.bin", &areaOptMap31},
        // {"database_area_opt_onlyCano/areaOptMap32.bin", &areaOptMap32},
        // {"database_area_opt_onlyCano/areaOptMap33.bin", &areaOptMap33},
        {"database_area_opt_haveNonCano/areaOptMap41.bin", &areaOptMap41}
        // {"database_area_opt_onlyCano/areaOptMap42.bin", &areaOptMap42},
        // {"database_area_opt_onlyCano/areaOptMap43.bin", &areaOptMap43}
    };

    for (const auto& task : tasks) {
        loadBinMMap(task.filename, *task.target);
    }
}

// index must be sorted before use this function!
double Database::getWindowArea(const std::vector<ll>& index, ll nVars, ll nOuts) const {
    const std::map<std::vector<ll>, double>* target = nullptr;
    if (nVars == 2) {
        if (nOuts == 2)
            target = &map22;
        else if (nOuts == 3)
            target = &map23;
        else if (nOuts == 1)
            target = &map21;
        else
            assert(0);
    }
    else if (nVars == 3) {
        if (nOuts == 2)
            target = &map32;
        else if (nOuts == 3)
            target = &map33;
        else if (nOuts == 1)
            target = &map31;
        else
            assert(0);
    }
    else if (nVars == 4) {
        if (nOuts == 2)
            target = &map42;
        else if (nOuts == 3)
            target = &map43;
        else if (nOuts == 1)
            target = &map41;
        else
            assert(0);
    }
    else
        assert(0);

    // sort(index.begin(), index.end());
    auto it = target->find(index);
    if (it != target->end()) {
        return it->second;
    } 
    else {
        return -1;
    }
}

double Database::SearchSuppDB(const std::vector<ll>& index, ll nVars, ll nOuts) const {
    const std::map<std::vector<ll>, double>* target = nullptr;
    if (nVars == 3) {
        if (nOuts == 2)
            target = &suppMap32;
        else if (nOuts == 3)
            target = &suppMap33;
        else
            assert(0);
    }
    else if (nVars == 4) {
        if (nOuts == 2)
            target = &suppMap42;
        else if (nOuts == 3)
            target = &suppMap43;
        else
            assert(0);
    }
    else {
        // cout << "nVars = " << nVars << ", nOuts = " << nOuts << endl;
        // assert(0);
        return -1;
    }

    // sort(index.begin(), index.end());
    auto it = target->find(index);
    if (it != target->end()) {
        return it->second;
    } 
    else {
        return -1;
    }
}

std::vector<uint8_t> Database::getNpnPerm(ll index, ll nVars) const {
    const std::vector<NpnInfo>* vec = nullptr;
    if (nVars == 3) vec = &npn3;
    else if (nVars == 4) vec = &npn4;
    else {
        std::cerr << "Unsupported nVars: " << nVars << "\n";
        assert(0);
    }

    if (index >= vec->size()) {
        std::cerr << "Index out of range: " << index << "\n";
        assert(0);
    }

    const NpnInfo& info = (*vec)[index];
    std::vector<uint8_t> permutation(info.permutation, info.permutation + nVars);
    return permutation;
}

std::pair<uint8_t, uint8_t> Database::getNpnInv(ll index, int nVars) const {
    const std::vector<NpnInfo>* vec = nullptr;
    if (nVars == 3) vec = &npn3;
    else if (nVars == 4) vec = &npn4;
    else {
        std::cerr << "Unsupported nVars: " << nVars << "\n";
        assert(0);
    }

    if (index >= vec->size()) {
        std::cerr << "Index out of range: " << index << "\n";
        assert(0);
    }

    const NpnInfo& info = (*vec)[index];
    return {info.input_inversion, info.output_inversion};
}

uint16_t Database::getNpnCanoFunc(ll index, int nVars) const {
    const std::vector<NpnInfo>* vec = nullptr;
    if (nVars == 3) vec = &npn3;
    else if (nVars == 4) vec = &npn4;
    else {
        std::cerr << "Unsupported nVars: " << nVars << "\n";
        assert(0);
    }

    if (index >= vec->size()) {
        std::cerr << "Index out of range: " << index << "\n";
        assert(0);
    }

    return (*vec)[index].canoFunc;
}

bool isFuncAreaOpt(ll func, ll nVars) {
    if (nVars == 4)
        return areaOptCanoFuncs_4Input.count(func);
    else if (nVars == 3)
        return areaOptCanoFuncs_3Input.count(func);
    else if (nVars == 2)
        return areaOptFuncs_2Input.count(func);
    else
        assert(0);
}

void Database::InsertToMap(std::vector <ll> & key, double value, ll nVars, ll nOuts) {
    if (nVars == 2) {
        if (nOuts == 2)
            map22[key] = value;
        else if (nOuts == 3)
            map23[key] = value;
        else if (nOuts == 1)
            map21[key] = value;
        else
            assert(0);
    }
    else if (nVars == 3) {
        if (nOuts == 2)
            map32[key] = value;
        else if (nOuts == 3)
            map33[key] = value;
        else if (nOuts == 1)
            map31[key] = value;
        else
            assert(0);
    }
    else if (nVars == 4) {
        if (nOuts == 2)
            map42[key] = value;
        else if (nOuts == 3)
            map43[key] = value;
        else if (nOuts == 1)
            map41[key] = value;
        else
            assert(0);
    }
    else
        assert(0);
}

void Database::InsertToSuppMap(std::vector <ll> & key, double value, ll nVars, ll nOuts) {
    if (nVars == 3) {
        if (nOuts == 2)
            suppMap32[key] = value;
        else if (nOuts == 3)
            suppMap33[key] = value;
        else
            assert(0);
    }
    else if (nVars == 4) {
        if (nOuts == 2)
            suppMap42[key] = value;
        else if (nOuts == 3)
            suppMap43[key] = value;
        else
            assert(0);
    }
    else
        assert(0);
}

void Database::UpdateDB() {
    std::map<std::vector<ll>, double>* maps[5][4];
    maps[2][1] = &map21;
    maps[2][2] = &map22;
    maps[2][3] = &map23;

    maps[3][1] = &map31;
    maps[3][2] = &map32;
    maps[3][3] = &map33;

    maps[4][1] = &map41;
    maps[4][2] = &map42;
    maps[4][3] = &map43;

    for (ll nVars = 2; nVars <= 4; ++nVars) {
        for (ll nOuts = 1; nOuts <= 3; ++nOuts) {
            if (fUpdate[nVars][nOuts] == 1) {
                std::string csvFile = "database/map" + std::to_string(nVars) + std::to_string(nOuts) + ".csv";
                std::string binFile = "database/map" + std::to_string(nVars) + std::to_string(nOuts) + ".bin";
                saveToCSV(*maps[nVars][nOuts], csvFile);
                cout << "update " << csvFile << endl;
                saveToBin(*maps[nVars][nOuts], binFile);
                cout << "update " << binFile << endl;
            }
        }
    }
}

void Database::UpdateSuppDB() {
    std::map<std::vector<ll>, double>* maps[5][4];
    maps[3][2] = &suppMap32;
    maps[3][3] = &suppMap33;
    maps[4][2] = &suppMap42;
    maps[4][3] = &suppMap43;

    for (ll nVars = 3; nVars <= 4; ++nVars) {
        for (ll nOuts = 2; nOuts <= 3; ++nOuts) {
            if (fUpdate[nVars][nOuts] == 1) {
                std::string csvFile = "database_supp/suppMap" + std::to_string(nVars) + std::to_string(nOuts) + ".csv";
                std::string binFile = "database_supp/suppMap" + std::to_string(nVars) + std::to_string(nOuts) + ".bin";
                saveToCSV(*maps[nVars][nOuts], csvFile);
                cout << "update " << csvFile << endl;
                saveToBin(*maps[nVars][nOuts], binFile);
                cout << "update " << binFile << endl;
            }
        }
    }
}

void Database::CalcAvg() {
    std::map<std::vector<ll>, double>* maps[5][4];
    maps[2][1] = &map21;
    maps[2][2] = &map22;
    maps[2][3] = &map23;

    maps[3][1] = &map31;
    maps[3][2] = &map32;
    maps[3][3] = &map33;

    maps[4][1] = &map41;
    maps[4][2] = &map42;
    maps[4][3] = &map43;

    for (ll nVars = 2; nVars <= 4; ++nVars) {
        for (ll nOuts = 1; nOuts <= 3; ++nOuts) {
            // double sum = 0;
            // for (const auto & pair : *maps[nVars][nOuts])
            //     sum += pair.second;
            // double avg = sum / (*maps[nVars][nOuts]).size();
            // cout << "avg area for map" << nVars << nOuts << ": " << avg << endl;

            const std::vector<double> bins = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
            std::map<std::pair<double, double>, int> histogram;       
            for (size_t i = 0; i < bins.size() - 1; ++i) {
                histogram[{bins[i], bins[i+1]}] = 0;
            }
            for (const auto& pair : *maps[nVars][nOuts]) {
                double value = pair.second;
                
                for (size_t i = 0; i < bins.size() - 1; ++i) {
                    if (value >= bins[i] && value < bins[i+1]) {
                        histogram[{bins[i], bins[i+1]}]++;
                        break;
                    }
                }
                if (value >= bins.back()) {
                    histogram[{bins[bins.size()-2], bins.back()}]++;
                }
            }
            cout << "values of map" << nVars << nOuts << ": " << endl;
            for (const auto& bin : histogram) {
                cout << "[" << bin.first.first << ", " << bin.first.second << "): " << bin.second << endl;
            }
            cout << endl;
        }
    }
}


void Database::GenDB_AreaOpt() {
    std::map<std::vector<ll>, double>* maps[5][4];
    maps[2][1] = &map21;
    maps[2][2] = &map22;
    maps[2][3] = &map23;

    maps[3][1] = &map31;
    maps[3][2] = &map32;
    maps[3][3] = &map33;

    maps[4][1] = &map41;
    maps[4][2] = &map42;
    maps[4][3] = &map43;

    double areaLim[5][4] = {
        {},   // i = 0 (all values are 0.0)
        {},
        {0.0, 1.2, 1.2, 1.2},   // i = 2, i.e. [2][0] ~ [2][3]  // choose max area for 3-input sub-circuit in experiments(1.18)
        {0.0, 1.7, 1.7, 1.7},   // i = 3, i.e. [3][0] ~ [3][3]  // choose max area for 4-input sub-circuit in experiments(1.67)    
        {0.0, 2.1, 2.1, 2.1},   // i = 4, i.e. [4][0] ~ [4][3]  // choose max area for 5-input sub-circuit in experiments(2.06)
    };

    std::vector<NpnInfo> * npnMap[5];
    npnMap[3] = &npn3;
    npnMap[4] = &npn4;

    for (ll nVars = 2; nVars <= 4; ++nVars) {
        for (ll nOuts = 1; nOuts <= 3; ++nOuts) {
            std::multimap<double, std::vector<ll>> areaOptMap;

            for (const auto& pair : *maps[nVars][nOuts]) {
                double value = pair.second;
                if (value > areaLim[nVars][nOuts])
                    continue;
                
                // if (nVars > 2) {    // consider non-canonical function combinations
                //     // obtain non-canonical functions
                //     vector <vector <ll>> nonCanoFuncs(nOuts);
                //     for (ll o = 0; o < nOuts; ++o) {
                //         ll canoFunc = pair.first[o];
                //         for (ll ii = 0; ii < (*npnMap[nVars]).size(); ++ii) {
                //             if ((*npnMap[nVars])[ii].canoFunc != canoFunc)
                //                 continue;
                //             nonCanoFuncs[o].push_back(ii);
                //         }
                //     }

                //     // obtain combinations of non-canonical functions
                //     if (nOuts == 1) {
                //         for (ll x : nonCanoFuncs[0]) {
                //             vector <ll> func = {x};
                //             auto dataPair = SynthFunction(x, nVars);
                //             double area = dataPair.first;
                //             if (area > areaLim[nVars][nOuts])
                //                 continue;
                //             areaOptMap.insert({area, func});
                //         }
                //     }
                //     else if (nOuts == 2) {
                //         for (ll x : nonCanoFuncs[0]) {
                //             for (ll y : nonCanoFuncs[1]) {
                //                 if (y == x)
                //                     continue;
                //                 vector <ll> funcs = {x, y};
                //                 sort(funcs.begin(), funcs.end());
                //                 auto area = SynthFunction_MultiOut(funcs, nVars);
                //                 if (area > areaLim[nVars][nOuts])
                //                     continue;
                //                 areaOptMap.insert({area, funcs});
                //             }
                //         }
                //     }
                //     else if (nOuts == 3) {
                //         for (ll x : nonCanoFuncs[0]) {
                //             for (ll y : nonCanoFuncs[1]) {
                //                 if (y == x)
                //                     continue;
                //                 for (ll z : nonCanoFuncs[2]) {
                //                     if (z == x || z == y)
                //                         continue;
                //                     vector <ll> funcs = {x, y, z};
                //                     sort(funcs.begin(), funcs.end());
                //                     auto area = SynthFunction_MultiOut(funcs, nVars);
                //                     if (area > areaLim[nVars][nOuts])
                //                         continue;
                //                     areaOptMap.insert({area, funcs});
                //                 }
                //             }
                //         }
                //     }                  
                // }
                // else {
                //     // nVars = 2, no need to obtain non-canonical functions again
                //     areaOptMap.insert({value, pair.first});
                // }
                areaOptMap.insert({value, pair.first});
            }

            std::string csvFile = "database_area_opt/areaOptMap" + std::to_string(nVars) + std::to_string(nOuts) + ".csv";
            std::string binFile = "database_area_opt/areaOptMap" + std::to_string(nVars) + std::to_string(nOuts) + ".bin";
            saveMultimapToCSV(areaOptMap, csvFile);
            saveMultimapToBin(areaOptMap, binFile);
            cout << "finish gen areaOptMap" << nVars << nOuts << endl;
        }
    }


}

void saveMultimapToBin(const std::multimap<double, std::vector<ll>>& mmap, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << '\n';
        return;
    }

    // Write multimap size
    size_t mapSize = mmap.size();
    file.write(reinterpret_cast<const char*>(&mapSize), sizeof(mapSize));

    for (const auto& [key, vec] : mmap) {
        // Write double key
        file.write(reinterpret_cast<const char*>(&key), sizeof(key));

        // Write vector size
        size_t vecSize = vec.size();
        file.write(reinterpret_cast<const char*>(&vecSize), sizeof(vecSize));

        // Write vector elements
        file.write(reinterpret_cast<const char*>(vec.data()), vecSize * sizeof(ll));
    }

    file.close();
}

void saveMultimapToCSV(const std::multimap<double, std::vector<ll>>& mmap, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << '\n';
        return;
    }

    for (const auto& [key, vec] : mmap) {
        file << key; // write double key
        for (ll val : vec) {
            file << "," << val; // comma-separated vector elements
        }
        file << "\n";
    }

    file.close();
}

void Database::loadBinMMap(const std::string& filename, std::multimap<double, std::vector<ll>>& table) {
    table.clear(); // discard previous contents
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << '\n';
        return;
    }

    size_t mapSize;
    file.read(reinterpret_cast<char*>(&mapSize), sizeof(mapSize));

    for (size_t i = 0; i < mapSize; ++i) {
        double key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));

        size_t vecSize;
        file.read(reinterpret_cast<char*>(&vecSize), sizeof(vecSize));

        std::vector<ll> vec(vecSize);
        file.read(reinterpret_cast<char*>(vec.data()), vecSize * sizeof(ll));

        table.emplace(key, std::move(vec));
    }

    file.close();
}

void Database::GenDB_nonCanoFunc() {
    map<ll, vector<ll>> nonCanoMap3;
    for (ll i = 0; i < npn3.size(); ++i) {
        ll canoFunc = npn3[i].canoFunc;
        auto it = nonCanoMap3.find(canoFunc);
        if (it != nonCanoMap3.end())
            it->second.push_back(i);
        else
            nonCanoMap3[canoFunc] = {i};
    }

    map<ll, vector<ll>> nonCanoMap4;
    for (ll i = 0; i < npn4.size(); ++i) {
        ll canoFunc = npn4[i].canoFunc;
        auto it = nonCanoMap4.find(canoFunc);
        if (it != nonCanoMap4.end())
            it->second.push_back(i);
        else
            nonCanoMap4[canoFunc] = {i};
    }

    saveMapToCSV(nonCanoMap3, "database/nonCanoMap3.csv");
    saveMapToBin(nonCanoMap3, "database/nonCanoMap3.bin");
    saveMapToCSV(nonCanoMap4, "database/nonCanoMap4.csv");
    saveMapToBin(nonCanoMap4, "database/nonCanoMap4.bin");
}

// chunked
void saveMapToCSV(const std::map<ll, std::vector<ll>>& table, const std::string& filename) {
    size_t chunkSize = 20;
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << '\n';
        return;
    }

    for (const auto& [key, vec] : table) {
        size_t total = vec.size();
        for (size_t i = 0; i < total; i += chunkSize) {
            file << key;
            size_t end = std::min(i + chunkSize, total);
            for (size_t j = i; j < end; ++j) {
                file << "," << vec[j];
            }
            file << "\n";
        }
    }

    file.close();
}

void saveMapToBin(const std::map<ll, std::vector<ll>>& table, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << '\n';
        return;
    }

    size_t mapSize = table.size();
    file.write(reinterpret_cast<const char*>(&mapSize), sizeof(mapSize));

    for (const auto& [key, vec] : table) {
        // Write key
        file.write(reinterpret_cast<const char*>(&key), sizeof(key));

        // Write vector size
        size_t vecSize = vec.size();
        file.write(reinterpret_cast<const char*>(&vecSize), sizeof(vecSize));

        // Write vector elements
        file.write(reinterpret_cast<const char*>(vec.data()), vecSize * sizeof(ll));
    }

    file.close();
}

void Database::loadBinMap2(const std::string& filename, std::map<ll, std::vector<ll>>& table) {
    table.clear();
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << '\n';
        return;
    }

    size_t mapSize;
    file.read(reinterpret_cast<char*>(&mapSize), sizeof(mapSize));

    for (size_t i = 0; i < mapSize; ++i) {
        ll key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));

        size_t vecSize;
        file.read(reinterpret_cast<char*>(&vecSize), sizeof(vecSize));

        std::vector<ll> vec(vecSize);
        file.read(reinterpret_cast<char*>(vec.data()), vecSize * sizeof(ll));

        table[key] = std::move(vec);
    }

    file.close();
}

void Database::loadNonCanoDB() {
    loadBinMap2("database/nonCanoMap3.bin", nonCanoMap3);
    loadBinMap2("database/nonCanoMap4.bin", nonCanoMap4);
}


double Database::GetAreaOptObj(ll nVars, ll nOuts, ll index, std::vector<ll> & funcs) {
    std::multimap<double, std::vector<ll>>* maps[5][4];
    maps[2][1] = &areaOptMap21;
    maps[2][2] = &areaOptMap22;
    maps[2][3] = &areaOptMap23;

    maps[3][1] = &areaOptMap31;
    maps[3][2] = &areaOptMap32;
    maps[3][3] = &areaOptMap33;

    maps[4][1] = &areaOptMap41;
    maps[4][2] = &areaOptMap42;
    maps[4][3] = &areaOptMap43;

    auto it = std::next(maps[nVars][nOuts]->begin(), index);
    funcs = it->second;
    return it->first;
}

ll Database::GetMMapSize(ll nVars, ll nOuts) {
    std::multimap<double, std::vector<ll>>* maps[5][4];
    maps[2][1] = &areaOptMap21;
    maps[2][2] = &areaOptMap22;
    maps[2][3] = &areaOptMap23;

    maps[3][1] = &areaOptMap31;
    maps[3][2] = &areaOptMap32;
    maps[3][3] = &areaOptMap33;

    maps[4][1] = &areaOptMap41;
    maps[4][2] = &areaOptMap42;
    maps[4][3] = &areaOptMap43;

    return maps[nVars][nOuts]->size();
}

void Database::SearchCands(ll nVars, ll nOuts, const std::vector <ll> & vIniFeasibleTt, double accArea, const std::vector <std::vector <ll>> & hamMarks, std::vector <Cand> & cands, ll nIniFlipBits, Simulator & appSmlt, BdData & bdData, ll hdTh, ll maxAppRWNum, std::set <ll> hdRank) {
    if (!(nVars == 2 || nVars == 3 || nVars == 4)) {
        cout << "nVars = " << nVars << endl;
        assert(0);
    }

    std::multimap<double, std::vector<ll>>* areaOptMap[5][4];
    areaOptMap[2][1] = &areaOptMap21;
    areaOptMap[2][2] = &areaOptMap22;
    areaOptMap[2][3] = &areaOptMap23;

    areaOptMap[3][1] = &areaOptMap31;
    areaOptMap[3][2] = &areaOptMap32;
    areaOptMap[3][3] = &areaOptMap33;

    areaOptMap[4][1] = &areaOptMap41;
    areaOptMap[4][2] = &areaOptMap42;
    areaOptMap[4][3] = &areaOptMap43;

    // double areaLim = accArea - 0.2;
    double areaLim = accArea;
    // double areaLim2 = accArea - 0.5;
    double areaLim2 = accArea;

    // Here hdTh has been subtracted by nIniFlipBits, but for hdRank's value, should add it.  

    if (nOuts == 3) {
        if (nVars == 2) {
            // (1) 3 different funcs
            for (const auto& [area, funcVec] : areaOptMap23) {
                if (area >= areaLim)
                    break;
                
                vector<ll> funcPerm = funcVec;
                do {
                    ll flipNum = CalcFlipBitNum(funcPerm, vIniFeasibleTt, hamMarks, hdTh);
                    if (flipNum > hdTh)
                        continue;
                    
                    cands.emplace_back(area, funcPerm, flipNum);
                    // update hdRank & hdTh (local, only update in this function)
                    hdRank.insert(flipNum + nIniFlipBits);
                    if (hdRank.size() >= maxAppRWNum) {
                        auto it = hdRank.begin();
                        std::advance(it, maxAppRWNum - 1);
                        hdTh = (*it) - nIniFlipBits;
                    }
                } while (std::next_permutation(funcPerm.begin(), funcPerm.end()));
            }
            // (2) 2 different funcs
            for (const auto& [area, funcVec] : areaOptMap22) {
                if (area >= areaLim)
                    break;
                
                vector<ll> funcPerm;
                for (ll i = 0; i < 2; ++i) {
                    funcPerm = funcVec;
                    funcPerm.push_back(funcVec[i]);
                    sort(funcPerm.begin(), funcPerm.end());
                    do {
                        ll flipNum = CalcFlipBitNum(funcPerm, vIniFeasibleTt, hamMarks, hdTh);
                        if (flipNum > hdTh)
                            continue;
                        cands.emplace_back(area, funcPerm, flipNum);
                        // update hdRank & hdTh (local, only update in this function)
                        hdRank.insert(flipNum + nIniFlipBits);
                        if (hdRank.size() >= maxAppRWNum) {
                            auto it = hdRank.begin();
                            std::advance(it, maxAppRWNum - 1);
                            hdTh = (*it) - nIniFlipBits;
                        }
                    } while (std::next_permutation(funcPerm.begin(), funcPerm.end()));
                }
            }
            // (3) 3 funcs are the same
            for (const auto& [area, funcVec] : areaOptMap21) {
                if (area >= areaLim)
                    break;
                
                vector<ll> funcPerm = {funcVec[0], funcVec[0], funcVec[0]};
                ll flipNum = CalcFlipBitNum(funcPerm, vIniFeasibleTt, hamMarks, hdTh);
                if (flipNum > hdTh)
                    continue;
                cands.emplace_back(area, funcPerm, flipNum);   
                // update hdRank & hdTh (local, only update in this function)
                hdRank.insert(flipNum + nIniFlipBits);
                if (hdRank.size() >= maxAppRWNum) {
                    auto it = hdRank.begin();
                    std::advance(it, maxAppRWNum - 1);
                    hdTh = (*it) - nIniFlipBits;
                }    
            }
        }
        else {
            // (1) 3 different funcs
            for (const auto& [area, funcVec] : *areaOptMap[nVars][3]) {
                if (area >= areaLim2)
                    break;
                
                vector<ll> funcPerm = funcVec;
                do {
                    // search non-cano funcs
                    vector <vector<ll>> vFuncs(3);
                    vector <vector<ll>> vFuncsHd(3);
                    for (ll i = 0; i < 3; ++i) 
                        SearchNonCanoFuncPro(nVars, funcPerm[i], vIniFeasibleTt[i], hamMarks[i], vFuncs[i], hdTh, vFuncsHd[i]);
                    for (ll x = 0; x < vFuncs[0].size(); ++x) {
                        ll totalHd = vFuncsHd[0][x];
                        if (totalHd > hdTh)
                            break;
                        for (ll y = 0; y < vFuncs[1].size(); ++y) {
                            totalHd += vFuncsHd[1][y];
                            if (totalHd > hdTh)
                                break;
                            for (ll z = 0; z < vFuncs[2].size(); ++z) {
                                totalHd += vFuncsHd[2][z];
                                if (totalHd > hdTh) 
                                    break;
                                vector <ll> funcNew = {vFuncs[0][x], vFuncs[1][y], vFuncs[2][z]};
                                auto funcNewBackUp = funcNew;

                                // check area
                                sort(funcNew.begin(), funcNew.end());
                                double areaNew = SearchSuppDB(funcNew, nVars, 3);
                                if (areaNew == -1) {
                                    areaNew = SynthFunction_MultiOut(funcNew, nVars);
                                    {
                                        std::lock_guard<std::mutex> lock(suppDbMutex);
                                        // add data to online db
                                        InsertToSuppMap(funcNew, areaNew, nVars, 3);
                                        // mark for updating the offline db
                                        SetfUpdate(nVars, 3);
                                    }
                                }

                                if (areaNew >= areaLim2)
                                    continue;
                                cands.emplace_back(areaNew, funcNewBackUp, totalHd);
                                // update hdRank & hdTh (local, only update in this function)
                                hdRank.insert(totalHd + nIniFlipBits);
                                if (hdRank.size() >= maxAppRWNum) {
                                    auto it = hdRank.begin();
                                    std::advance(it, maxAppRWNum - 1);
                                    hdTh = (*it) - nIniFlipBits;
                                }
                            }
                        }
                    }
                } while (std::next_permutation(funcPerm.begin(), funcPerm.end()));
            }
            // (2) 2 different funcs
            for (const auto& [area, funcVec] : *areaOptMap[nVars][2]) {
                if (area >= areaLim2)
                    break;
                
                vector<ll> funcPerm;
                for (ll i = 0; i < 2; ++i) {
                    funcPerm = funcVec;
                    funcPerm.push_back(funcVec[i]);
                    sort(funcPerm.begin(), funcPerm.end());

                    do {
                        // search non-cano funcs
                        vector <vector<ll>> vFuncs(3);
                        vector <vector<ll>> vFuncsHd(3);
                        for (ll i = 0; i < 3; ++i) 
                            SearchNonCanoFuncPro(nVars, funcPerm[i], vIniFeasibleTt[i], hamMarks[i], vFuncs[i], hdTh, vFuncsHd[i]);
                        for (ll x = 0; x < vFuncs[0].size(); ++x) {
                            ll totalHd = vFuncsHd[0][x];
                            if (totalHd > hdTh)
                                break;
                            for (ll y = 0; y < vFuncs[1].size(); ++y) {
                                totalHd += vFuncsHd[1][y];
                                if (totalHd > hdTh)
                                    break;
                                for (ll z = 0; z < vFuncs[2].size(); ++z) {
                                    totalHd += vFuncsHd[2][z];
                                    if (totalHd > hdTh) 
                                        break;
                                    vector <ll> funcNew = {vFuncs[0][x], vFuncs[1][y], vFuncs[2][z]};
                                    auto funcNewBackUp = funcNew;
                                    
                                    // check area
                                    sort(funcNew.begin(), funcNew.end());
                                    double areaNew = SearchSuppDB(funcNew, nVars, 3);
                                    if (areaNew == -1) {
                                        areaNew = SynthFunction_MultiOut(funcNew, nVars);
                                        {
                                            std::lock_guard<std::mutex> lock(suppDbMutex);
                                            // add data to online db
                                            InsertToSuppMap(funcNew, areaNew, nVars, 3);
                                            // mark for updating the offline db
                                            SetfUpdate(nVars, 3);
                                        }
                                    }
                                        
                                    if (areaNew >= areaLim)
                                        continue;
                                    cands.emplace_back(areaNew, funcNewBackUp, totalHd);
                                    // update hdRank & hdTh (local, only update in this function)
                                    hdRank.insert(totalHd + nIniFlipBits);
                                    if (hdRank.size() >= maxAppRWNum) {
                                        auto it = hdRank.begin();
                                        std::advance(it, maxAppRWNum - 1);
                                        hdTh = (*it) - nIniFlipBits;
                                    }
                                }
                            }
                        }
                    } while (std::next_permutation(funcPerm.begin(), funcPerm.end()));
                }
            }
            // (3) 3 funcs are the same
            // for (const auto& [area, funcVec] : *areaOptMap[nVars][1]) {
            //     if (area >= areaLim)
            //         break;
                
            //     vector<ll> funcPerm = {funcVec[0], funcVec[0], funcVec[0]};

            //     vector <vector<ll>> vFuncs(3);
            //     for (ll i = 0; i < 3; ++i) 
            //         SearchNonCanoFunc(nVars, funcPerm[i], vIniFeasibleTt[i], hamMarks[i], vFuncs[i]);
            //     for (ll x : vFuncs[0]) {
            //         for (ll y : vFuncs[1]) {
            //             for (ll z : vFuncs[2]) {
            //                 vector <ll> funcNew = {x, y, z};
            //                 // look up area data
            //                 vector <ll> funcNew_forSearch = funcNew;
            //                 sort(funcNew_forSearch.begin(), funcNew_forSearch.end());
            //                 double areaNew = GetMMapArea(nVars, funcNew_forSearch);
            //                 if (areaNew == -1 || areaNew >= accArea)
            //                     break;
            //                 ll flipNum = CalcFlipBitNum(funcNew, vIniFeasibleTt, hamMarks);
            //                 cands.emplace_back(areaNew, funcNew, flipNum);
            //             }
            //         }
            //     }       
            // }
            for (const auto& [area, funcVec] : *areaOptMap[nVars][1]) {
                if (area >= areaLim)
                    break;
                
                vector<ll> funcPerm = {funcVec[0], funcVec[0], funcVec[0]};
                ll flipNum = CalcFlipBitNum(funcPerm, vIniFeasibleTt, hamMarks, hdTh);
                if (flipNum > hdTh)
                    continue;
                cands.emplace_back(area, funcPerm, flipNum);
                // update hdRank & hdTh (local, only update in this function)
                hdRank.insert(flipNum + nIniFlipBits);
                if (hdRank.size() >= maxAppRWNum) {
                    auto it = hdRank.begin();
                    std::advance(it, maxAppRWNum - 1);
                    hdTh = (*it) - nIniFlipBits;
                }      
            }
        }
    }
    else if (nOuts == 2) {
        if (nVars == 2) {
            // (1) 2 different funcs
            for (const auto& [area, funcVec] : areaOptMap22) {
                if (area >= areaLim)
                    break;
                
                vector<ll> funcPerm = funcVec;
                do {
                    ll flipNum = CalcFlipBitNum(funcPerm, vIniFeasibleTt, hamMarks, hdTh);
                    if (flipNum > hdTh)
                        continue;
                    cands.emplace_back(area, funcPerm, flipNum);
                    // update hdRank & hdTh (local, only update in this function)
                    hdRank.insert(flipNum + nIniFlipBits);
                    if (hdRank.size() >= maxAppRWNum) {
                        auto it = hdRank.begin();
                        std::advance(it, maxAppRWNum - 1);
                        hdTh = (*it) - nIniFlipBits;
                    }
                } while (std::next_permutation(funcPerm.begin(), funcPerm.end()));
            }
            // (2) 2 funcs are the same
            for (const auto& [area, funcVec] : areaOptMap21) {
                if (area >= areaLim)
                    break;
                
                vector<ll> funcPerm = {funcVec[0], funcVec[0]};
                ll flipNum = CalcFlipBitNum(funcPerm, vIniFeasibleTt, hamMarks, hdTh);
                if (flipNum > hdTh)
                    continue;
                cands.emplace_back(area, funcPerm, flipNum);  
                // update hdRank & hdTh (local, only update in this function)
                hdRank.insert(flipNum + nIniFlipBits);
                if (hdRank.size() >= maxAppRWNum) {
                    auto it = hdRank.begin();
                    std::advance(it, maxAppRWNum - 1);
                    hdTh = (*it) - nIniFlipBits;
                }    
            }
        }
        else {
            // (1) 2 different funcs
            for (const auto& [area, funcVec] : *areaOptMap[nVars][2]) {
                if (area >= areaLim2)
                    break;
                
                vector<ll> funcPerm = funcVec;
                do {
                    // search non-cano funcs
                    vector <vector<ll>> vFuncs(2);
                    vector <vector<ll>> vFuncsHd(2);
                    for (ll i = 0; i < 2; ++i) 
                        SearchNonCanoFuncPro(nVars, funcPerm[i], vIniFeasibleTt[i], hamMarks[i], vFuncs[i], hdTh, vFuncsHd[i]);
                    for (ll x = 0; x < vFuncs[0].size(); ++x) {
                        ll totalHd = vFuncsHd[0][x];
                        if (totalHd > hdTh)
                            break;
                        for (ll y = 0; y < vFuncs[1].size(); ++y) {
                            totalHd += vFuncsHd[1][y];
                            if (totalHd > hdTh) 
                                break;
                            vector <ll> funcNew = {vFuncs[0][x], vFuncs[1][y]};
                            auto funcNewBackUp = funcNew;
                            
                            // check area
                            sort(funcNew.begin(), funcNew.end());
                            double areaNew = SearchSuppDB(funcNew, nVars, 2);
                            if (areaNew == -1) {
                                areaNew = SynthFunction_MultiOut(funcNew, nVars);
                                {
                                    std::lock_guard<std::mutex> lock(suppDbMutex);
                                    // add data to online db
                                    InsertToSuppMap(funcNew, areaNew, nVars, 2);
                                    // mark for updating the offline db
                                    SetfUpdate(nVars, 2);
                                }
                            }

                            if (areaNew >= areaLim)
                                continue;
                            cands.emplace_back(areaNew, funcNewBackUp, totalHd);
                            // update hdRank & hdTh (local, only update in this function)
                            hdRank.insert(totalHd + nIniFlipBits);
                            if (hdRank.size() >= maxAppRWNum) {
                                auto it = hdRank.begin();
                                std::advance(it, maxAppRWNum - 1);
                                hdTh = (*it) - nIniFlipBits;
                            }
                        }
                    }
                } while (std::next_permutation(funcPerm.begin(), funcPerm.end()));
            }
            // (2) 2 funcs are the same
            // for (const auto& [area, funcVec] : *areaOptMap[nVars][1]) {
            //     if (area >= areaLim)
            //         break;
                
            //     vector<ll> funcPerm = {funcVec[0], funcVec[0]};

            //     vector <vector<ll>> vFuncs(2);
            //     for (ll i = 0; i < 2; ++i) 
            //         SearchNonCanoFunc(nVars, funcPerm[i], vIniFeasibleTt[i], hamMarks[i], vFuncs[i]);
            //     for (ll x : vFuncs[0]) {
            //         for (ll y : vFuncs[1]) {
            //             vector <ll> funcNew = {x, y};
            //             // look up area data
            //             vector <ll> funcNew_forSearch = funcNew;
            //             sort(funcNew_forSearch.begin(), funcNew_forSearch.end());
            //             double areaNew = GetMMapArea(nVars, funcNew_forSearch);
            //             if (areaNew == -1 || areaNew >= accArea)
            //                 break;
            //             ll flipNum = CalcFlipBitNum(funcNew, vIniFeasibleTt, hamMarks);
            //             cands.emplace_back(areaNew, funcNew, flipNum);
            //         }
            //     }       
            // }
            for (const auto& [area, funcVec] : *areaOptMap[nVars][1]) {
                if (area >= areaLim)
                    break;
                
                vector<ll> funcPerm = {funcVec[0], funcVec[0]};
                ll flipNum = CalcFlipBitNum(funcPerm, vIniFeasibleTt, hamMarks, hdTh);
                if (flipNum > hdTh)
                    continue;
                cands.emplace_back(area, funcPerm, flipNum);   
                // update hdRank & hdTh (local, only update in this function)
                hdRank.insert(flipNum + nIniFlipBits);
                if (hdRank.size() >= maxAppRWNum) {
                    auto it = hdRank.begin();
                    std::advance(it, maxAppRWNum - 1);
                    hdTh = (*it) - nIniFlipBits;
                }   
            }
        }
    }
    else if (nOuts == 1) {
        // if (nVars == 2) {
        //     for (const auto& [area, funcVec] : areaOptMap21) {
        //         if (area >= areaLim)
        //             break;
                
        //         ll flipNum = CalcFlipBitNum(funcVec, vIniFeasibleTt, hamMarks);
        //         cands.emplace_back(area, funcVec, flipNum);        
        //     }
        // }
        // else {
        //     for (const auto& [area, funcVec] : *areaOptMap[nVars][1]) {
        //         if (area >= areaLim)
        //             break;
                
        //         vector<ll> funcPerm = {funcVec[0]};

        //         vector <vector<ll>> vFuncs(1);
        //         SearchNonCanoFunc(nVars, funcPerm[0], vIniFeasibleTt[0], hamMarks[0], vFuncs[0]);
        //         for (ll x : vFuncs[0]) {
        //             vector <ll> funcNew = {x};
        //             // look up area data
        //             double areaNew = GetMMapArea(nVars, funcNew);
        //             if (areaNew == -1 || areaNew >= accArea)
        //                 break;
        //             ll flipNum = CalcFlipBitNum(funcNew, vIniFeasibleTt, hamMarks);
        //             cands.emplace_back(areaNew, funcNew, flipNum);
        //         }       
        //     }
        // }
        for (const auto& [area, funcVec] : *areaOptMap[nVars][1]) {
            if (area >= areaLim)
                break;
            
            ll flipNum = CalcFlipBitNum(funcVec, vIniFeasibleTt, hamMarks, hdTh);
            if (flipNum > hdTh)
                continue;
            cands.emplace_back(area, funcVec, flipNum);   
            // update hdRank & hdTh (local, only update in this function)
            hdRank.insert(flipNum + nIniFlipBits);
            if (hdRank.size() >= maxAppRWNum) {
                auto it = hdRank.begin();
                std::advance(it, maxAppRWNum - 1);
                hdTh = (*it) - nIniFlipBits;
            }   
        }
    }
    else {
        cout << "nOuts = " << nOuts << endl;
        assert(0);
    }
}


void Database::SearchCandsByAppFuncLib(ll nVars, ll nOuts, const std::vector <ll> & vIniFeasibleTt, double accArea, const std::vector <std::vector <ll>> & hamMarks, std::vector <Cand> & cands, ll nIniFlipBits, Simulator & appSmlt, ll hdTh, ll maxAppRWNum, std::set <ll> hdRank) {
    if (!(nVars == 2 || nVars == 3 || nVars == 4)) {
        cout << "nVars = " << nVars << endl;
        assert(0);
    }
    std::vector < std::vector <ll> >* appFuncMap = nullptr;
    switch (nVars) {
        case 2:
            appFuncMap = &appFuncMap2;
            break;
        case 3:
            appFuncMap = &appFuncMap3;
            break;
        case 4:
            appFuncMap = &appFuncMap4;
            break;
    }

    vector < vector < pair <ll, ll> > > appFuncs(vIniFeasibleTt.size());   // <func, hd>
    ll iLo = 0;
    assert(nOuts == vIniFeasibleTt.size());
    for (const auto & f0: vIniFeasibleTt) {
        assert(f0 < appFuncMap->size());
        for (const auto & f : (*appFuncMap)[f0]) {
            double area = SearchAreaMap(f, nVars);
            if (area == -1|| area > accArea)
                continue;
            ll hd = CalcFlipBitNum_singleLO(f, f0, hamMarks[iLo], -1);
            appFuncs[iLo].emplace_back(f, hd);
        }
        ++iLo;
    }

    for (ll i = 0; i < nOuts; ++i) {
        std::sort(appFuncs[i].begin(), appFuncs[i].end(), [](const pair <ll, ll> & a, const pair <ll, ll> & b) {
            return a.second < b.second;
        });
    }

    // Here hdTh has been subtracted by nIniFlipBits, but for hdRank's value, should add it.  

    if (nOuts == 2) {
        for (ll i = 0; i < appFuncs[0].size(); ++i) {
            ll hd = appFuncs[0][i].second;
            if (hd > hdTh)
                break;
            double area0 = SearchAreaMap(appFuncs[0][i].first, nVars);
            
            for (ll j = 0; j < appFuncs[1].size(); ++j) {
                hd += appFuncs[1][j].second;
                if (hd > hdTh)
                    break;
                double area1 = SearchAreaMap(appFuncs[1][j].first, nVars);
                vector <ll> fApp = {appFuncs[0][i].first, appFuncs[1][j].first};

                // check area
                double areaNew = -1;
                if (fApp[0] == fApp[1]) {
                    assert(area0 == area1);
                    areaNew = area0;
                }
                else {
                    auto fAppSort = fApp;
                    std::sort(fAppSort.begin(), fAppSort.end());
                    areaNew = SearchSuppDB(fAppSort, nVars, nOuts);
                    // if (areaNew == -1) {
                    //     areaNew = SynthFunction_MultiOut(fAppSort, nVars);
                    //     {
                    //         std::lock_guard<std::mutex> lock(suppDbMutex);
                    //         // add data to online db
                    //         InsertToSuppMap(fAppSort, areaNew, nVars, nOuts);
                    //         // mark for updating the offline db
                    //         SetfUpdate(nVars, nOuts);
                    //     }
                    // }
                }
                if (areaNew >= accArea)
                    continue;
                cands.emplace_back(areaNew, fApp, hd + nIniFlipBits);
                // update hdRank & hdTh (local, only update in this function)
                hdRank.insert(hd + nIniFlipBits);
                if (hdRank.size() >= maxAppRWNum) {
                    auto it = hdRank.begin();
                    std::advance(it, maxAppRWNum - 1);
                    hdTh = (*it) - nIniFlipBits;
                }
            }
        }
    }
    else if (nOuts == 3) {
        for (ll i = 0; i < appFuncs[0].size(); ++i) {
            ll hd = appFuncs[0][i].second;
            if (hd > hdTh)
                break;
            double area0 = SearchAreaMap(appFuncs[0][i].first, nVars);
            
            for (ll j = 0; j < appFuncs[1].size(); ++j) {
                hd += appFuncs[1][j].second;
                if (hd > hdTh)
                    break;
                double area1 = SearchAreaMap(appFuncs[1][j].first, nVars);
                
                for (ll k = 0; k < appFuncs[2].size(); ++k) {
                    hd += appFuncs[2][k].second;
                    if (hd > hdTh)
                        break;
                    double area2 = SearchAreaMap(appFuncs[0][i].first, nVars);
                    
                    vector <ll> fApp = {appFuncs[0][i].first, appFuncs[1][j].first, appFuncs[2][k].first};

                    // check area
                    auto fAppSort = fApp;
                    sort(fAppSort.begin(), fAppSort.end());
                    auto last = std::unique(fAppSort.begin(), fAppSort.end());
                    fAppSort.erase(last, fAppSort.end());
                    double areaNew = -1;
                    if (fAppSort.size() == 1) {
                        assert(area0 == area1);
                        assert(area1 == area2);
                        areaNew = area0;
                    }
                    else {
                        areaNew = SearchSuppDB(fAppSort, nVars, fAppSort.size());
                        // if (areaNew == -1) {
                        //     areaNew = SynthFunction_MultiOut(fAppSort, nVars);
                        //     {
                        //         std::lock_guard<std::mutex> lock(suppDbMutex);
                        //         // add data to online db
                        //         InsertToSuppMap(fAppSort, areaNew, nVars, fAppSort.size());
                        //         // mark for updating the offline db
                        //         SetfUpdate(nVars, fAppSort.size());
                        //     }
                        // }
                    }

                    if (areaNew >= accArea)
                        continue;
                    cands.emplace_back(areaNew, fApp, hd + nIniFlipBits);
                    // update hdRank & hdTh (local, only update in this function)
                    hdRank.insert(hd + nIniFlipBits);
                    if (hdRank.size() >= maxAppRWNum) {
                        auto it = hdRank.begin();
                        std::advance(it, maxAppRWNum - 1);
                        hdTh = (*it) - nIniFlipBits;
                    }
                }
            }
        }
    }
    else {
        cout << "nOuts = " << nOuts << endl;
        assert(0);
    }
}


void Database::SearchCandsPro(ll nVars, ll nOuts, std::vector <ll> vIniFeasibleTt, double accArea, std::vector <std::vector <ll>> & hamMarks, std::vector <Cand> & cands, ll nIniFlipBits, Simulator & appSmlt, BdData & bdData, METR_TYPE metrType, const vector <ll> & vDiv, const vector <ll> & vLO, ll vLoId) {
    if (!(nVars == 2 || nVars == 3 || nVars == 4)) {
        cout << "nVars = " << nVars << endl;
        assert(0);
    }

    std::multimap<double, std::vector<ll>>* areaOptMap[5][4];
    areaOptMap[2][1] = &areaOptMap21;
    areaOptMap[2][2] = &areaOptMap22;
    areaOptMap[2][3] = &areaOptMap23;

    areaOptMap[3][1] = &areaOptMap31;
    areaOptMap[3][2] = &areaOptMap32;
    areaOptMap[3][3] = &areaOptMap33;

    areaOptMap[4][1] = &areaOptMap41;
    areaOptMap[4][2] = &areaOptMap42;
    areaOptMap[4][3] = &areaOptMap43;

    // ll nCand = 10;
    ll nCand = 30; 
    double areaLim = accArea - 0.2;
    // double areaLim = accArea;
    double areaLim2 = accArea - 0.5;
    // double areaLim2 = accArea;
    double runMin = std::numeric_limits<double>::max();

    if (nOuts == 3) {
        if (nVars == 2) {
            // (1) 3 different funcs
            for (const auto& [area, funcVec] : areaOptMap23) {
                if (area >= areaLim)
                    break;
                
                vector<ll> funcPerm = funcVec;
                do {
                    double flipScore = CalcFlipBitScore(funcPerm, runMin, appSmlt, bdData, vDiv, vLO, vLoId, metrType);
                    if (flipScore > runMin)
                        continue;
                    cands.emplace_back(area, funcPerm, 0, flipScore);
                    if (cands.size() > nCand) {     // update runMin
                        assert(cands.size() == nCand + 1);
                        sort(cands.begin(), cands.end(), [](const Cand & A, const Cand & B) {
                            if (A.score != B.score)
                                return A.score < B.score;
                            return A.area < B.area;
                        });
                        cands.pop_back();
                        runMin = cands.end()->score;
                    }
                } while (std::next_permutation(funcPerm.begin(), funcPerm.end()));
            }
            // (2) 2 different funcs
            for (const auto& [area, funcVec] : areaOptMap22) {
                if (area >= areaLim)
                    break;
                
                vector<ll> funcPerm;
                for (ll i = 0; i < 2; ++i) {
                    funcPerm = funcVec;
                    funcPerm.push_back(funcVec[i]);
                    sort(funcPerm.begin(), funcPerm.end());
                    do {
                        double flipScore = CalcFlipBitScore(funcPerm, runMin, appSmlt, bdData, vDiv, vLO, vLoId, metrType);
                        if (flipScore > runMin)
                            continue;
                        cands.emplace_back(area, funcPerm, 0, flipScore);
                        if (cands.size() > nCand) {     // update runMin
                            assert(cands.size() == nCand + 1);
                            sort(cands.begin(), cands.end(), [](const Cand & A, const Cand & B) {
                                if (A.score != B.score)
                                    return A.score < B.score;
                                return A.area < B.area;
                            });
                            cands.pop_back();
                            runMin = cands.end()->score;
                        }
                    } while (std::next_permutation(funcPerm.begin(), funcPerm.end()));
                }
            }
            // (3) 3 funcs are the same
            for (const auto& [area, funcVec] : areaOptMap21) {
                if (area >= areaLim)
                    break;
                
                vector<ll> funcPerm = {funcVec[0], funcVec[0], funcVec[0]};
                double flipScore = CalcFlipBitScore(funcPerm, runMin, appSmlt, bdData, vDiv, vLO, vLoId, metrType);
                if (flipScore > runMin)
                    continue;
                cands.emplace_back(area, funcPerm, 0, flipScore);   
                if (cands.size() > nCand) {     // update runMin
                    assert(cands.size() == nCand + 1);
                    sort(cands.begin(), cands.end(), [](const Cand & A, const Cand & B) {
                        if (A.score != B.score)
                            return A.score < B.score;
                        return A.area < B.area;
                    });
                    cands.pop_back();
                    runMin = cands.end()->score;
                }     
            }
        }
        else {
            // (1) 3 different funcs
            for (const auto& [area, funcVec] : *areaOptMap[nVars][3]) {
                if (area >= areaLim2)
                    break;
                
                vector<ll> funcPerm = funcVec;
                do {
                    // search non-cano funcs
                    vector <vector<ll>> vFuncs(3);
                    for (ll i = 0; i < 3; ++i) 
                        SearchNonCanoFunc(nVars, funcPerm[i], vIniFeasibleTt[i], hamMarks[i], vFuncs[i], runMin);
                    for (ll x : vFuncs[0]) {
                        for (ll y : vFuncs[1]) {
                            for (ll z : vFuncs[2]) {
                                vector <ll> funcNew = {x, y, z};
                                double flipScore = CalcFlipBitScore(funcNew, runMin, appSmlt, bdData, vDiv, vLO, vLoId, metrType);
                                if (flipScore > runMin)
                                    continue;
                                
                                // check area
                                sort(funcNew.begin(), funcNew.end());
                                double areaNew = SearchSuppDB(funcNew, nVars, 3);
                                if (areaNew == -1) {
                                    areaNew = SynthFunction_MultiOut(funcNew, nVars);
                                    {
                                        std::lock_guard<std::mutex> lock(suppDbMutex);
                                        // add data to online db
                                        InsertToSuppMap(funcNew, areaNew, nVars, 3);
                                        // mark for updating the offline db
                                        SetfUpdate(nVars, 3);
                                    }
                                }

                                if (areaNew >= areaLim2)
                                    continue;
                                cands.emplace_back(areaNew, funcNew, 0, flipScore);
                                if (cands.size() > nCand) {     // update runMin
                                    assert(cands.size() == nCand + 1);
                                    sort(cands.begin(), cands.end(), [](const Cand & A, const Cand & B) {
                                        if (A.score != B.score)
                                            return A.score < B.score;
                                        return A.area < B.area;
                                    });
                                    cands.pop_back();
                                    runMin = cands.end()->score;
                                }
                            }
                        }
                    }
                } while (std::next_permutation(funcPerm.begin(), funcPerm.end()));
            }
            // (2) 2 different funcs
            for (const auto& [area, funcVec] : *areaOptMap[nVars][2]) {
                if (area >= areaLim2)
                    break;
                
                vector<ll> funcPerm;
                for (ll i = 0; i < 2; ++i) {
                    funcPerm = funcVec;
                    funcPerm.push_back(funcVec[i]);
                    sort(funcPerm.begin(), funcPerm.end());

                    do {
                        // search non-cano funcs
                        vector <vector<ll>> vFuncs(3);
                        for (ll i = 0; i < 3; ++i) 
                            SearchNonCanoFunc(nVars, funcPerm[i], vIniFeasibleTt[i], hamMarks[i], vFuncs[i], runMin);
                        for (ll x : vFuncs[0]) {
                            for (ll y : vFuncs[1]) {
                                for (ll z : vFuncs[2]) {
                                    vector <ll> funcNew = {x, y, z};
                                    double flipScore = CalcFlipBitScore(funcNew, runMin, appSmlt, bdData, vDiv, vLO, vLoId, metrType);
                                    if (flipScore > runMin)
                                        continue;
                                    
                                    // check area
                                    sort(funcNew.begin(), funcNew.end());
                                    double areaNew = SearchSuppDB(funcNew, nVars, 3);
                                    if (areaNew == -1) {
                                        areaNew = SynthFunction_MultiOut(funcNew, nVars);
                                        {
                                            std::lock_guard<std::mutex> lock(suppDbMutex);
                                            // add data to online db
                                            InsertToSuppMap(funcNew, areaNew, nVars, 3);
                                            // mark for updating the offline db
                                            SetfUpdate(nVars, 3);
                                        }
                                    }
                                        
                                    if (areaNew >= areaLim)
                                        continue;
                                    cands.emplace_back(areaNew, funcNew, 0, flipScore);
                                    if (cands.size() > nCand) {     // update runMin
                                        assert(cands.size() == nCand + 1);
                                        sort(cands.begin(), cands.end(), [](const Cand & A, const Cand & B) {
                                            if (A.score != B.score)
                                                return A.score < B.score;
                                            return A.area < B.area;
                                        });
                                        cands.pop_back();
                                        runMin = cands.end()->score;
                                    }
                                }
                            }
                        }
                    } while (std::next_permutation(funcPerm.begin(), funcPerm.end()));
                }
            }
            // (3) 3 funcs are the same
            // for (const auto& [area, funcVec] : *areaOptMap[nVars][1]) {
            //     if (area >= areaLim)
            //         break;
                
            //     vector<ll> funcPerm = {funcVec[0], funcVec[0], funcVec[0]};

            //     vector <vector<ll>> vFuncs(3);
            //     for (ll i = 0; i < 3; ++i) 
            //         SearchNonCanoFunc(nVars, funcPerm[i], vIniFeasibleTt[i], hamMarks[i], vFuncs[i]);
            //     for (ll x : vFuncs[0]) {
            //         for (ll y : vFuncs[1]) {
            //             for (ll z : vFuncs[2]) {
            //                 vector <ll> funcNew = {x, y, z};
            //                 // look up area data
            //                 vector <ll> funcNew_forSearch = funcNew;
            //                 sort(funcNew_forSearch.begin(), funcNew_forSearch.end());
            //                 double areaNew = GetMMapArea(nVars, funcNew_forSearch);
            //                 if (areaNew == -1 || areaNew >= accArea)
            //                     break;
            //                 ll flipNum = CalcFlipBitNum(funcNew, vIniFeasibleTt, hamMarks);
            //                 cands.emplace_back(areaNew, funcNew, flipNum);
            //             }
            //         }
            //     }       
            // }
            for (const auto& [area, funcVec] : *areaOptMap[nVars][1]) {
                if (area >= areaLim)
                    break;
                
                vector<ll> funcPerm = {funcVec[0], funcVec[0], funcVec[0]};
                double flipScore = CalcFlipBitScore(funcPerm, runMin, appSmlt, bdData, vDiv, vLO, vLoId, metrType);
                if (flipScore > runMin)
                    continue;
                cands.emplace_back(area, funcPerm, 0, flipScore);
                if (cands.size() > nCand) {     // update runMin
                    assert(cands.size() == nCand + 1);
                    sort(cands.begin(), cands.end(), [](const Cand & A, const Cand & B) {
                        if (A.score != B.score)
                            return A.score < B.score;
                        return A.area < B.area;
                    });
                    cands.pop_back();
                    runMin = cands.end()->score;
                }        
            }
        }
    }
    else if (nOuts == 2) {
        if (nVars == 2) {
            // (1) 2 different funcs
            for (const auto& [area, funcVec] : areaOptMap22) {
                if (area >= areaLim)
                    break;
                
                vector<ll> funcPerm = funcVec;
                do {
                    double flipScore = CalcFlipBitScore(funcPerm, runMin, appSmlt, bdData, vDiv, vLO, vLoId, metrType);
                    if (flipScore > runMin)
                        continue;
                    cands.emplace_back(area, funcPerm, 0, flipScore);
                    if (cands.size() > nCand) {     // update runMin
                        assert(cands.size() == nCand + 1);
                        sort(cands.begin(), cands.end(), [](const Cand & A, const Cand & B) {
                            if (A.score != B.score)
                                return A.score < B.score;
                            return A.area < B.area;
                        });
                        cands.pop_back();
                        runMin = cands.end()->score;
                    }
                } while (std::next_permutation(funcPerm.begin(), funcPerm.end()));
            }
            // (2) 2 funcs are the same
            for (const auto& [area, funcVec] : areaOptMap21) {
                if (area >= areaLim)
                    break;
                
                vector<ll> funcPerm = {funcVec[0], funcVec[0]};
                double flipScore = CalcFlipBitScore(funcPerm, runMin, appSmlt, bdData, vDiv, vLO, vLoId, metrType);
                if (flipScore > runMin)
                    continue;
                cands.emplace_back(area, funcPerm, 0, flipScore);  
                if (cands.size() > nCand) {     // update runMin
                    assert(cands.size() == nCand + 1);
                    sort(cands.begin(), cands.end(), [](const Cand & A, const Cand & B) {
                        if (A.score != B.score)
                            return A.score < B.score;
                        return A.area < B.area;
                    });
                    cands.pop_back();
                    runMin = cands.end()->score;
                }      
            }
        }
        else {
            // (1) 2 different funcs
            for (const auto& [area, funcVec] : *areaOptMap[nVars][2]) {
                if (area >= areaLim2)
                    break;
                
                vector<ll> funcPerm = funcVec;
                do {
                    // search non-cano funcs
                    vector <vector<ll>> vFuncs(2);
                    for (ll i = 0; i < 2; ++i) 
                        SearchNonCanoFunc(nVars, funcPerm[i], vIniFeasibleTt[i], hamMarks[i], vFuncs[i], runMin);
                    for (ll x : vFuncs[0]) {
                        for (ll y : vFuncs[1]) {
                            vector <ll> funcNew = {x, y};
                            double flipScore = CalcFlipBitScore(funcNew, runMin, appSmlt, bdData, vDiv, vLO, vLoId, metrType);
                            if (flipScore > runMin)
                                continue;
                            
                            // check area
                            sort(funcNew.begin(), funcNew.end());
                            double areaNew = SearchSuppDB(funcNew, nVars, 2);
                            if (areaNew == -1) {
                                areaNew = SynthFunction_MultiOut(funcNew, nVars);
                                {
                                    std::lock_guard<std::mutex> lock(suppDbMutex);
                                    // add data to online db
                                    InsertToSuppMap(funcNew, areaNew, nVars, 2);
                                    // mark for updating the offline db
                                    SetfUpdate(nVars, 2);
                                }
                            }

                            if (areaNew >= areaLim)
                                continue;
                            cands.emplace_back(areaNew, funcNew, 0, flipScore);
                            if (cands.size() > nCand) {     // update runMin
                                assert(cands.size() == nCand + 1);
                                sort(cands.begin(), cands.end(), [](const Cand & A, const Cand & B) {
                                    if (A.score != B.score)
                                        return A.score < B.score;
                                    return A.area < B.area;
                                });
                                cands.pop_back();
                                runMin = cands.end()->score;
                            }
                        }
                    }
                } while (std::next_permutation(funcPerm.begin(), funcPerm.end()));
            }
            // (2) 2 funcs are the same
            // for (const auto& [area, funcVec] : *areaOptMap[nVars][1]) {
            //     if (area >= areaLim)
            //         break;
                
            //     vector<ll> funcPerm = {funcVec[0], funcVec[0]};

            //     vector <vector<ll>> vFuncs(2);
            //     for (ll i = 0; i < 2; ++i) 
            //         SearchNonCanoFunc(nVars, funcPerm[i], vIniFeasibleTt[i], hamMarks[i], vFuncs[i]);
            //     for (ll x : vFuncs[0]) {
            //         for (ll y : vFuncs[1]) {
            //             vector <ll> funcNew = {x, y};
            //             // look up area data
            //             vector <ll> funcNew_forSearch = funcNew;
            //             sort(funcNew_forSearch.begin(), funcNew_forSearch.end());
            //             double areaNew = GetMMapArea(nVars, funcNew_forSearch);
            //             if (areaNew == -1 || areaNew >= accArea)
            //                 break;
            //             ll flipNum = CalcFlipBitNum(funcNew, vIniFeasibleTt, hamMarks);
            //             cands.emplace_back(areaNew, funcNew, flipNum);
            //         }
            //     }       
            // }
            for (const auto& [area, funcVec] : *areaOptMap[nVars][1]) {
                if (area >= areaLim)
                    break;
                
                vector<ll> funcPerm = {funcVec[0], funcVec[0]};
                double flipScore = CalcFlipBitScore(funcPerm, runMin, appSmlt, bdData, vDiv, vLO, vLoId, metrType);
                if (flipScore > runMin)
                    continue;
                cands.emplace_back(area, funcPerm, 0, flipScore);   
                if (cands.size() > nCand) {     // update runMin
                    assert(cands.size() == nCand + 1);
                    sort(cands.begin(), cands.end(), [](const Cand & A, const Cand & B) {
                        if (A.score != B.score)
                            return A.score < B.score;
                        return A.area < B.area;
                    });
                    cands.pop_back();
                    runMin = cands.end()->score;
                }     
            }
        }
    }
    else if (nOuts == 1) {
        // if (nVars == 2) {
        //     for (const auto& [area, funcVec] : areaOptMap21) {
        //         if (area >= areaLim)
        //             break;
                
        //         ll flipNum = CalcFlipBitNum(funcVec, vIniFeasibleTt, hamMarks);
        //         cands.emplace_back(area, funcVec, flipNum);        
        //     }
        // }
        // else {
        //     for (const auto& [area, funcVec] : *areaOptMap[nVars][1]) {
        //         if (area >= areaLim)
        //             break;
                
        //         vector<ll> funcPerm = {funcVec[0]};

        //         vector <vector<ll>> vFuncs(1);
        //         SearchNonCanoFunc(nVars, funcPerm[0], vIniFeasibleTt[0], hamMarks[0], vFuncs[0]);
        //         for (ll x : vFuncs[0]) {
        //             vector <ll> funcNew = {x};
        //             // look up area data
        //             double areaNew = GetMMapArea(nVars, funcNew);
        //             if (areaNew == -1 || areaNew >= accArea)
        //                 break;
        //             ll flipNum = CalcFlipBitNum(funcNew, vIniFeasibleTt, hamMarks);
        //             cands.emplace_back(areaNew, funcNew, flipNum);
        //         }       
        //     }
        // }
        for (const auto& [area, funcVec] : *areaOptMap[nVars][1]) {
            if (area >= areaLim)
                break;
            
            double flipScore = CalcFlipBitScore(funcVec, runMin, appSmlt, bdData, vDiv, vLO, vLoId, metrType);
            if (flipScore > runMin)
                continue;
            cands.emplace_back(area, funcVec, 0, flipScore);   
            if (cands.size() > nCand) {     // update runMin
                assert(cands.size() == nCand + 1);
                sort(cands.begin(), cands.end(), [](const Cand & A, const Cand & B) {
                    if (A.score != B.score)
                        return A.score < B.score;
                    return A.area < B.area;
                });
                cands.pop_back();
                runMin = cands.end()->score;
            }     
        }
    }
    else {
        cout << "nOuts = " << nOuts << endl;
        assert(0);
    }

    if (cands.empty())
        return;

    // sort by score
    if (cands.size() > nCand) {
        std::nth_element(cands.begin(), cands.begin() + nCand, cands.end(), 
            [](const Cand & A, const Cand & B) {return A.score < B.score;}
        );
        cands.resize(nCand);
    }
    sort(cands.begin(), cands.end(), [](const Cand & A, const Cand & B) {
        return A.score < B.score;
    });
}




ll CalcFlipBitNum(std::vector <ll> newFunc, std::vector <ll> refFunc, const std::vector <std::vector <ll>> & hamMarks, ll limit) {
    ll sum = 0;
    assert(newFunc.size() == refFunc.size());
    for (ll o = 0; o < refFunc.size(); ++o) {
        ll diff = newFunc[o] ^ refFunc[o];
        for (ll iPatt = 0; iPatt < hamMarks[0].size(); ++iPatt) {
            if ((diff >> iPatt) & 1) {
                sum += hamMarks[o][iPatt];
                if (sum > limit)
                    return sum;
            }
        }
    }
    return sum;
}

ll CalcFlipBitNum_singleLO(ll newFunc, ll refFunc, const std::vector <ll> & hamMark, ll limit) {
    ll sum = 0;

    ll diff = newFunc ^ refFunc;
    for (ll iPatt = 0; iPatt < hamMark.size(); ++iPatt) {
        if ((diff >> iPatt) & 1) {
            sum += hamMark[iPatt];
            if (limit != -1 && sum > limit)
                return sum;
        }
    }

    return sum;
}

void Database::SearchNonCanoFunc(ll nVars, ll canoFunc, ll refFunc, const std::vector <ll> & hamMark, std::vector <ll> & vNonCanoFuncs, ll hamLimit) {
    std::map<ll, std::vector<ll>> * nonCanoMap;
    if (nVars == 3)
        nonCanoMap = &nonCanoMap3;
    else if (nVars == 4)
        nonCanoMap = &nonCanoMap4;
    else
        assert(0);
    
    ll limit = 5;   
    vector <pair<ll, ll>> Funcs;

    // consider all non-canonical funcs
    auto it = (*nonCanoMap).find(canoFunc);
    if (it == (*nonCanoMap).end()) {
        cout << "nVars = " << nVars << ", canoFunc = " << canoFunc << endl;
        assert(it != (*nonCanoMap).end());
    }
    const std::vector<ll>& vec = it->second;
    for (ll nonCanoFunc : vec) {
        ll nFlip = CalcFlipBitNum_singleLO(nonCanoFunc, refFunc, hamMark, hamLimit);
        if (nFlip > hamLimit)
            continue;
        Funcs.emplace_back(nonCanoFunc, nFlip);
    }
    // add canoFunc
    ll nFlip = CalcFlipBitNum_singleLO(canoFunc, refFunc, hamMark, hamLimit);
    if (nFlip <= hamLimit)
        Funcs.emplace_back(canoFunc, nFlip);

    // sort
    if (Funcs.size() > limit) {
        nth_element(Funcs.begin(), Funcs.begin() + limit, Funcs.end(), [](const auto & A, const auto & B) {
            return A.second < B.second;
        });
        Funcs.resize(limit);
        sort(Funcs.begin(), Funcs.end());
    }
    else {
        sort(Funcs.begin(), Funcs.end());
    }
    
    for (ll i = 0; i < Funcs.size(); ++i) {
        vNonCanoFuncs.push_back(Funcs[i].first);
    }
}

void Database::SearchNonCanoFuncPro(ll nVars, ll canoFunc, ll refFunc, const std::vector <ll> & hamMark, std::vector <ll> & vNonCanoFuncs, ll hamLimit, std::vector <ll> & vNonCanoFuncsHd) {
    std::map<ll, std::vector<ll>> * nonCanoMap;
    if (nVars == 3)
        nonCanoMap = &nonCanoMap3;
    else if (nVars == 4)
        nonCanoMap = &nonCanoMap4;
    else
        assert(0);
    
    ll limit = 5;   
    vector <pair<ll, ll>> Funcs;

    // consider all non-canonical funcs
    auto it = (*nonCanoMap).find(canoFunc);
    if (it == (*nonCanoMap).end()) {
        cout << "nVars = " << nVars << ", canoFunc = " << canoFunc << endl;
        assert(it != (*nonCanoMap).end());
    }
    const std::vector<ll>& vec = it->second;
    for (ll nonCanoFunc : vec) {
        ll nFlip = CalcFlipBitNum_singleLO(nonCanoFunc, refFunc, hamMark, hamLimit);
        if (nFlip > hamLimit)
            continue;
        Funcs.emplace_back(nonCanoFunc, nFlip);
    }
    // add canoFunc
    ll nFlip = CalcFlipBitNum_singleLO(canoFunc, refFunc, hamMark, hamLimit);
    if (nFlip <= hamLimit)
        Funcs.emplace_back(canoFunc, nFlip);

    // sort
    if (Funcs.size() > limit) {
        nth_element(Funcs.begin(), Funcs.begin() + limit, Funcs.end(), [](const auto & A, const auto & B) {
            return A.second < B.second;
        });
        Funcs.resize(limit);
        sort(Funcs.begin(), Funcs.end());
    }
    else {
        sort(Funcs.begin(), Funcs.end());
    }
    
    for (ll i = 0; i < Funcs.size(); ++i) {
        vNonCanoFuncs.push_back(Funcs[i].first);
        vNonCanoFuncsHd.push_back(Funcs[i].second);
    }
}


void Database::ObserveNonCanoFuncArea(ll canoFunc) {
    std::map<ll, std::vector<ll>> * nonCanoMap = &nonCanoMap4;
    auto dataPair = SynthFunction(canoFunc, 4);
    cout << "canoFunc = " << canoFunc << ", area = " << dataPair.first << endl;

    auto it = (*nonCanoMap).find(canoFunc);
    if (it == (*nonCanoMap).end()) {
        assert(it != (*nonCanoMap).end());
    }

    const std::vector<ll>& vec = it->second;
    for (ll nonCanoFunc : vec) {
        auto dataPair = SynthFunction(nonCanoFunc, 4);
        cout << nonCanoFunc << ": " << dataPair.first << endl;
    }
}

double Database::GetMMapArea(ll nVars, std::vector <ll> func) {
    std::multimap<double, std::vector<ll>> * areaOptMap;
    if (nVars == 3)
        areaOptMap = &areaOptMap31;
    else if (nVars == 4)
        areaOptMap = &areaOptMap41;
    else
        assert(0);

    for (const auto& [key, value] : *areaOptMap) {
        if (value == func) {
            return key;           
        }
    }

    return -1;
}

double CalcFlipBitScore(std::vector <ll> newFunc, ll runMin, Simulator & appSmlt, BdData & bdData, const vector <ll> & vDiv, const vector <ll> & vLO, ll vLoId, METR_TYPE metrType) {
    double score = 0;
    ll nPo = appSmlt.GetPoNum();

    for (ll iFrame = 0; iFrame < appSmlt.GetFrameNumb(); ++iFrame) {
        // get divisor pattern
        ll divPatt = 0; 
        for (ll i = 0; i < vDiv.size(); ++i) {
            divPatt = divPatt * 2 + appSmlt.GetDat(vDiv[i], iFrame);
        }

        // check whether each LO flips
        vector <int> flipMark(vLO.size(), 0);
        bool fFlip = false;
        for (ll o = 0; o < vLO.size(); ++o) {   // traverse LO
            if (appSmlt.GetDat(vLO[o], iFrame) != ((newFunc[o] >> divPatt) & 1)) {
                flipMark[o] = 1;
                fFlip = true;
            }
        }

        if (fFlip) {    // check bd
            const vector < vector < boost::dynamic_bitset <ull> > > * targetBd = nullptr;
            ll targetId = 0;
            if (vLO.size() == 2) {
                if (flipMark[0] && flipMark[1]) {
                    targetBd = & bdData.bdPo2Nodes11;
                    targetId = vLoId;
                }
                else if (flipMark[0] && (!flipMark[1])) {
                    if (bdData.vLO2Relation[vLoId] == 1) {
                        targetBd = & bdData.bdPo2Nodes10;
                        targetId = vLoId;
                    }
                    else {
                        targetBd = & bdData.bdPo2NodesRef;
                        targetId = vLO[0];
                    }
                }
                else if (!flipMark[0] && flipMark[1]) {
                    targetBd = & bdData.bdPo2NodesRef;
                    targetId = vLO[1];
                }
                else
                    assert(0);
            }
            else if (vLO.size() == 3) {
                if (flipMark[0] && flipMark[1] && flipMark[2]) {
                    targetBd = & bdData.bdPo2Nodes111;
                    targetId = vLoId;
                }
                else if (flipMark[0] && (!flipMark[1]) && flipMark[2]) {
                    targetBd = & bdData.bdPo2Nodes101;
                    targetId = vLoId;
                }
                else if (flipMark[0] && flipMark[1] && (!flipMark[2])) {
                    targetBd = & bdData.bdPo2Nodes110;
                    targetId = vLoId;
                }
                else if ((!flipMark[0]) && flipMark[1] && flipMark[2]) {
                    targetBd = & bdData.bdPo2Nodes011;
                    targetId = vLoId;
                }
                else if (flipMark[0] && (!flipMark[1]) && (!flipMark[2])) {
                    targetBd = & bdData.bdPo2NodesRef;
                    targetId = vLO[0];
                }
                else if ((!flipMark[0]) && flipMark[1] && (!flipMark[2])) {
                    targetBd = & bdData.bdPo2NodesRef;
                    targetId = vLO[1];
                }
                else if ((!flipMark[0]) && (!flipMark[1]) && flipMark[2]) {
                    targetBd = & bdData.bdPo2NodesRef;
                    targetId = vLO[2];
                }
                else
                    assert(0);
            }

            // calculate score
            if (metrType == METR_TYPE::MHD) {
                ll hd = 0;
                for (ll o = 0; o < nPo; ++o) {
                    if ((*targetBd)[o][targetId][iFrame])
                        hd += 1;
                }
                score += double(hd)/double(nPo);
            }
            else if (metrType == METR_TYPE::ER) {
                bool fErrPatt = false;
                for (ll o = 0; o < nPo; ++o) {
                    if ((*targetBd)[o][targetId][iFrame]) {
                        fErrPatt = true;
                        break;
                    }
                }
                score += fErrPatt ? 1 : 0;
            }
            else
                assert(0);  // haven't implemented yet for other metric types
        }
    }

    return score;
}

void Database::GenAppFuncDB(ll hdTh) {
    loadMMapDB();
    std::multimap<double, std::vector<ll>>* maps[5];
    maps[0] = nullptr;
    maps[1] = nullptr;
    maps[2] = &areaOptMap21;
    maps[3] = &areaOptMap31;
    maps[4] = &areaOptMap41;

    std::vector < std::vector <ll> > appFuncMap2;
    std::vector < std::vector <ll> > appFuncMap3;
    std::vector < std::vector <ll> > appFuncMap4;

    for (int i = 2; i <= 4; ++i) {
        if (maps[i] == nullptr)
            continue;
        std::cout << "Traversing map at index " << i << std::endl;

        std::vector < std::vector <ll> >* targetMap = nullptr;
        std::string filename;
        switch (i) {
            case 2:
                targetMap = &appFuncMap2;
                filename = "appFuncMap2";
                break;
            case 3:
                targetMap = &appFuncMap3;
                filename = "appFuncMap3";
                break;
            case 4:
                targetMap = &appFuncMap4;
                filename = "appFuncMap4";
                break;
            default:
                continue; 
        }

        ll tableBitNum = 1LL << i;
        ll maxFunc = 1LL << tableBitNum;
        targetMap->resize(maxFunc);
        for (ll accFunc = 0; accFunc < maxFunc; ++accFunc) {
            for (auto it = maps[i]->begin(); it != maps[i]->end(); ++it) {
                ll appFunc = it->second[0];
                if (accFunc == appFunc)
                    break;
                ll hd = std::popcount(static_cast<unsigned long long>(accFunc ^ appFunc));
                if (hd <= hdTh) {
                    (*targetMap)[accFunc].push_back(appFunc);
                }
            }
        }
        saveAppFuncDB(*targetMap, filename, 20);
    }   
}

void saveAppFuncDB(const std::vector<std::vector<ll>>& data, const std::string& filename, int maxValsPerLine) {
    if (data.empty()) {
        std::cerr << "Error: The input vector is empty." << std::endl;
        return;
    }

    std::filesystem::path dir_path = "database_appFunc";
    CreatePath(dir_path);
    std::string binFilename = (dir_path / (filename + ".bin")).string();
    std::string csvFilename = (dir_path / (filename + ".csv")).string();

    // Save to a .bin file (binary format, no line length risk)
    // std::string binFilename = filename + ".bin";
    std::ofstream binFile(binFilename, std::ios::binary);
    if (!binFile) {
        throw std::runtime_error("Could not open .bin file for writing: " + binFilename);
    }
    size_t rows = data.size();
    binFile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    for (size_t i = 0; i < rows; ++i) {
        size_t cols = data[i].size();
        binFile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        if (cols > 0) {
            binFile.write(reinterpret_cast<const char*>(data[i].data()), cols * sizeof(ll));
        }
    }
    binFile.close();
    std::cout << "Data successfully saved to " << binFilename << std::endl;

    // Save to a .csv file (text format, with controlled line length and index marking)
    // std::string csvFilename = filename + ".csv";
    std::ofstream csvFile(csvFilename);
    if (!csvFile) {
        throw std::runtime_error("Could not open .csv file for writing: " + csvFilename);
    }
    for (size_t i = 0; i < data.size(); ++i) {
        // Write the original index at the beginning of each line
        csvFile << i;
        int count = 0;
        
        // Iterate through all values in the inner vector
        for (const auto& val : data[i]) {
            csvFile << "," << val;
            count++;
            
            // If the number of elements per line reaches the limit, force a new line
            if (count >= maxValsPerLine) {
                // Write the index again at the start of the new line
                csvFile << "\n" << i;
                count = 0; // Reset the counter
            }
        }
        csvFile << "\n"; // End the current row
    }
    csvFile.close();
    std::cout << "Data successfully saved to " << csvFilename << std::endl;
}

// --- Function to load data from a .bin file ---
void Database::LoadAppFuncDB() {
    std::filesystem::path dir_path = "database_appFunc";
    for (ll i = 2; i <= 4; ++i) {
        std::vector < std::vector <ll> >* targetMap = nullptr;
        std::string filename;
        switch (i) {
            case 2:
                targetMap = &appFuncMap2;
                filename = "appFuncMap2";
                break;
            case 3:
                targetMap = &appFuncMap3;
                filename = "appFuncMap3";
                break;
            case 4:
                targetMap = &appFuncMap4;
                filename = "appFuncMap4";
                break;
            default:
                continue; 
        }
        
        std::string binFilename = (dir_path / (filename + ".bin")).string();
        std::ifstream binFile(binFilename, std::ios::binary);
        if (!binFile) {
            throw std::runtime_error("Could not open .bin file for reading: " + binFilename);
        }

        // Clear existing data
        targetMap->clear();
        
        // Read the total number of rows
        size_t rows;
        binFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        if (binFile.gcount() != sizeof(rows)) {
            throw std::runtime_error("Failed to read file header. The file may be corrupted.");
        }
        targetMap->resize(rows);

        // Loop through and read data for each row
        for (size_t i = 0; i < rows; ++i) {
            size_t cols;
            // Read the number of elements in the current row
            binFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));
            if (binFile.gcount() != sizeof(cols)) {
                throw std::runtime_error("Failed to read row size. The file may be corrupted.");
            }
            (*targetMap)[i].resize(cols);

            // If the row has data, read all the elements
            if (cols > 0) {
                binFile.read(reinterpret_cast<char*>((*targetMap)[i].data()), cols * sizeof(ll));
                if (binFile.gcount() != cols * sizeof(ll)) {
                    throw std::runtime_error("Failed to read row data. The file may be corrupted.");
                }
            }
        }
        binFile.close();
        std::cout << "Data successfully loaded from " << binFilename << "." << std::endl;
    }    
}

void Database::InitAreaMap() {
    for (const auto& pair : areaOptMap21) {
        areaMap2[pair.second[0]] = pair.first;  // Since the ll value is unique, it can be directly used as the key of the map
    }
    for (const auto& pair : areaOptMap31) {
        areaMap3[pair.second[0]] = pair.first;
    }
    for (const auto& pair : areaOptMap41) {
        areaMap4[pair.second[0]] = pair.first;
    }
} 

double Database::SearchAreaMap(ll func, ll nVars) {
    std::map<ll, double>* targetMap = nullptr;

    if (nVars == 2) {
        targetMap = &areaMap2;
    } else if (nVars == 3) {
        targetMap = &areaMap3;
    } else if (nVars == 4) {
        targetMap = &areaMap4;
    } else {
        cout << "nVars = " << nVars << endl;
        assert(0);
    }

    auto it = targetMap->find(func);

    if (it != targetMap->end()) {
        return it->second;
    } else {
        return -1;
    }
}


