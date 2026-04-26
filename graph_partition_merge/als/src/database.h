#pragma once

#include "header.h"
#include "lac.h"

extern const std::set<ll> areaOptFuncs_2Input;
extern const std::set<ll> areaOptCanoFuncs_3Input;
extern const std::set<ll> areaOptCanoFuncs_4Input;

struct NpnInfo {
    uint16_t canoFunc;
    uint8_t permutation[4];   // enough for nVars = 3 or 4
    uint8_t input_inversion;
    uint8_t output_inversion;
};

struct Cand {
    double area;
    std::vector <ll> appFuncs;
    ll nFlipBits;
    double score = 0.0;
};

struct BdData {
    const std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2Nodes11;
    const std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2Nodes10;
    const std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2Nodes101;
    const std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2Nodes110;
    const std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2Nodes011;
    const std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2Nodes111;
    const std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2NodesRef;
    const std::vector <int> & vLO2Relation;

    BdData(
        const std::vector < std::vector < boost::dynamic_bitset <ull> > > & p11,
        const std::vector < std::vector < boost::dynamic_bitset <ull> > > & p10,
        const std::vector < std::vector < boost::dynamic_bitset <ull> > > & p101,
        const std::vector < std::vector < boost::dynamic_bitset <ull> > > & p110,
        const std::vector < std::vector < boost::dynamic_bitset <ull> > > & p011,
        const std::vector < std::vector < boost::dynamic_bitset <ull> > > & p111,
        const std::vector < std::vector < boost::dynamic_bitset <ull> > > & pRef,
        const std::vector <int> & pLO2)
        : bdPo2Nodes11(p11), bdPo2Nodes10(p10), bdPo2Nodes101(p101),
          bdPo2Nodes110(p110), bdPo2Nodes011(p011), bdPo2Nodes111(p111), bdPo2NodesRef(pRef), vLO2Relation(pLO2) {}
};

class Database {
private:
    // 2 inputs
    std::map<std::vector<ll>, double> map21;    // elements in vector<ll> are stored from small to large
    std::map<std::vector<ll>, double> map22;
    std::map<std::vector<ll>, double> map23;
    // 3 inputs
    std::map<std::vector<ll>, double> map31;
    std::map<std::vector<ll>, double> map32;
    std::map<std::vector<ll>, double> map33;
    // 4 inputs
    std::map<std::vector<ll>, double> map41;
    std::map<std::vector<ll>, double> map42;
    std::map<std::vector<ll>, double> map43;
    // do not use the data above

    std::vector<NpnInfo> npn3;
    std::vector<NpnInfo> npn4;

    // 2 inputs
    std::multimap<double, std::vector<ll>> areaOptMap21;    // elements in vector<ll> are stored from small to large
    std::multimap<double, std::vector<ll>> areaOptMap22;
    std::multimap<double, std::vector<ll>> areaOptMap23;
    // 3 inputs
    std::multimap<double, std::vector<ll>> areaOptMap31;
    std::multimap<double, std::vector<ll>> areaOptMap32;
    std::multimap<double, std::vector<ll>> areaOptMap33;
    // 4 inputs
    std::multimap<double, std::vector<ll>> areaOptMap41;
    std::multimap<double, std::vector<ll>> areaOptMap42;
    std::multimap<double, std::vector<ll>> areaOptMap43;

    std::map<ll, std::vector<ll>> nonCanoMap3;
    std::map<ll, std::vector<ll>> nonCanoMap4;

    // Supplementary database (synth online to obtain)
    std::map<std::vector<ll>, double> suppMap32;
    std::map<std::vector<ll>, double> suppMap33;
    std::map<std::vector<ll>, double> suppMap42;
    std::map<std::vector<ll>, double> suppMap43;
    std::mutex suppDbMutex;

    // appFunc database
    std::vector < std::vector <ll> > appFuncMap2;
    std::vector < std::vector <ll> > appFuncMap3;
    std::vector < std::vector <ll> > appFuncMap4;

    // one-output function database
    std::map<ll, double> areaMap2;
    std::map<ll, double> areaMap3;
    std::map<ll, double> areaMap4;

    // data update flag
    int fUpdate[5][4] = {};     // nVars: {2, 3, 4}; nOuts: {1, 2, 3}

    // aig database
    

    bool loadBinNPN(const std::string& filepath, int nVars, std::vector<NpnInfo>& targetVec);
    void loadBinMap(const std::string& filename, std::map<std::vector<ll>, double>& table);

    void loadBinMMap(const std::string& filename, std::multimap<double, std::vector<ll>>& table);
    void loadBinMap2(const std::string& filename, std::map<ll, std::vector<ll>>& table);

public:
    void loadNpnDB();
    // void loadMapDB();
    void loadMMapDB();
    void loadNonCanoDB();
    void loadSuppDB();
    double getWindowArea(const std::vector<ll>& index, ll nVars, ll nOuts) const;     // Exm: db.getWindowArea(index, "map22");
    std::vector<uint8_t> getNpnPerm(ll index, ll nVars) const;
    std::pair<uint8_t, uint8_t> getNpnInv(ll index, int nVars) const;
    uint16_t getNpnCanoFunc(ll index, int nVars) const;
    void InsertToMap(std::vector <ll> & key, double value, ll nVars, ll nOuts);
    void InsertToSuppMap(std::vector <ll> & key, double value, ll nVars, ll nOuts);
    void SetfUpdate(ll nVars, ll nOuts) {fUpdate[nVars][nOuts] = 1;}
    void UpdateDB();
    void UpdateSuppDB();
    void CalcAvg();
    void GenDB_AreaOpt();
    void GenDB_nonCanoFunc();
    double GetAreaOptObj(ll nVars, ll nOuts, ll index, std::vector<ll> & funcs);
    ll GetMMapSize(ll nVars, ll nOuts);

    void SearchCands(ll nVars, ll nOuts, const std::vector <ll> & vIniFeasibleTt, double accArea, const std::vector <std::vector <ll>> & hamMarks, std::vector <Cand> & cands, ll nIniFlipBits, Simulator & appSmlt, BdData & bdData, ll hdTh, ll maxAppRWNum, std::set <ll> hdRank);
    void SearchCandsByAppFuncLib(ll nVars, ll nOuts, const std::vector <ll> & vIniFeasibleTt, double accArea, const std::vector <std::vector <ll>> & hamMarks, std::vector <Cand> & cands, ll nIniFlipBits, Simulator & appSmlt, ll hdTh, ll maxAppRWNum, std::set <ll> hdRank);
    void SearchCandsPro(ll nVars, ll nOuts, std::vector <ll> vIniFeasibleTt, double accArea, std::vector <std::vector <ll>> & hamMarks, std::vector <Cand> & cands, ll nIniFlipBits, Simulator & appSmlt, BdData & bdData, METR_TYPE metrType, const std::vector <ll> & vDiv, const std::vector <ll> & vLO, ll vLoId);
    void SearchNonCanoFunc(ll nVars, ll canoFunc, ll refFunc, const std::vector <ll> & hamMark, std::vector <ll> & vNonCanoFuncs, ll hamLimit);
    void SearchNonCanoFuncPro(ll nVars, ll canoFunc, ll refFunc, const std::vector <ll> & hamMark, std::vector <ll> & vNonCanoFuncs, ll hamLimit, std::vector <ll> & vNonCanoFuncsHd);
    double GetMMapArea(ll nVars, std::vector <ll> func);
    double SearchSuppDB(const std::vector<ll>& index, ll nVars, ll nOuts) const;
    void ObserveNonCanoFuncArea(ll canoFunc);

    void GenAppFuncDB(ll hdTh);
    void LoadAppFuncDB();
    void InitAreaMap();
    double SearchAreaMap(ll func, ll nVars);
};

void saveToCSV(const std::map<std::vector<ll>, double>& table, const std::string& filename);
void saveToBin(const std::map<std::vector<ll>, double>& table, const std::string& filename);
void saveMultimapToBin(const std::multimap<double, std::vector<ll>>& mmap, const std::string& filename);
void saveMultimapToCSV(const std::multimap<double, std::vector<ll>>& mmap, const std::string& filename);
void saveMapToCSV(const std::map<ll, std::vector<ll>>& table, const std::string& filename);
void saveMapToBin(const std::map<ll, std::vector<ll>>& table, const std::string& filename);

bool isFuncAreaOpt(ll func, ll nVars);
ll CalcFlipBitNum(std::vector <ll> newFunc, std::vector <ll> refFunc, const std::vector <std::vector <ll>> & hamMarks, ll limit);
ll CalcFlipBitNum_singleLO(ll newFunc, ll refFunc, const std::vector <ll> & hamMark, ll limit);
double CalcFlipBitScore(std::vector <ll> newFunc, ll runMin, Simulator & appSmlt, BdData & bdData, const std::vector <ll> & vDiv, const std::vector <ll> & vLO, ll vLoId, METR_TYPE metrType);
void saveAppFuncDB(const std::vector<std::vector<ll>>& data, const std::string& filename, int maxValsPerLine);

