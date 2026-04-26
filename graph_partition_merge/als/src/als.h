#pragma once


#include "header.h"
#include "my_abc.h"
#include "simulator.h"
#include "error.h"
#include "lac.h"
#include "database.h"
#include "subCkt.h"
#include <unordered_map>

struct ALSOpt {
    ll workMode;
    bool isSign;
    bool inpMapVerilog;
    unsigned sourceSeed;
    DISTR_TYPE distrType;
    METR_TYPE metrType;
    ll nFrame;
    ll nThread;
    ll nOutput;
    double errUppBound;
    std::string outpPath;
    abc::Abc_Ntk_t * pNtk;
    abc::Abc_Ntk_t * pAppNtk;
    std::string appCktName;
    std::string accCktName;
    
    void Print();
};

struct SubcktInfo {
    ll id;
    std::vector<ll> LO_ids;
    std::vector<ll> LI_ids;
    ll nodeNum;
    std::vector<ll> node_ids;
    std::string filename;
};

class ALSMan {
private:
    bool isSign;
    bool startFromAggrTrunc;
    bool propConst;
    // bool QuasiVert;
    bool useMultLac;
    // bool forbid2;
    // bool CompEnd;
    bool useAppDisjCut;
    unsigned sourceSeed;
    unsigned seed;
    LAC_TYPE lacType;
    DISTR_TYPE distrType;
    METR_TYPE metrType;
    ll nFrame;
    ll nOutput;
    ll nThread;
    const double errUppBound;
    double maxDelay;
    double maxArea;
    double maxDelayOri;
    NetMan accNet;
    boost::mt19937 randGen;
    // std::vector <ll> oriConstNodes;

    Database db;
    ll nCand;

    abc::Abc_Ntk_t * pAppNtk;
    bool measureMode;
    std::string appCktName;
    std::vector <std::vector <ll>> vLos;
    std::vector <SubcktInfo> vSubcktInfos;
    std::string outpPath;
    std::string accCktName;
    std::string subcktPath;


    ALSMan(const ALSMan &);
    ALSMan(ALSMan &&);
    ALSMan & operator = (const ALSMan &);
    ALSMan & operator = (ALSMan &&);

public:
    explicit ALSMan(ALSOpt & opt);
    ~ALSMan() = default;
    void GraphPartition();
    void GraphMerge();
    void GraphMerge_greedy();
    void GraphMerge_binary();
    double ReplaceSubcircuit(ll index, const std::vector<std::pair<ll, std::string>> & subcktFiles, const std::unordered_map<ll, SubcktInfo> & subcktInfoMap, NetMan & Net, bool fPrint = false);
    void ReplaceSubcircuit_v2(ll index, const std::vector<std::pair<ll, std::string>> & subcktFiles, const std::unordered_map<ll, SubcktInfo> & subcktInfoMap, NetMan & Net, bool fPrint = false);
    void GetSubckts_2LO(NetMan & net);  // Support only 2-output subgraphs
    void GetSubcktsPro(NetMan & net);  // Do not restrict the number of inputs or outputs of subgraphs; restrict the number of internal nodes instead.
    void OutputCurrSubckt(NetMan & net, std::vector <ll> & vLO, std::vector <ll> & vLI, ll subcktId, ll nodeNum);
    void ExtractSubckt(NetMan & net, std::vector <ll> & vLO, ll subcktId);
    void PrintSubcktInfos();
    ll FindPairLO(NetMan & net, ll id);
    
    unsigned NewSeed();
    bool VerErr(NetMan & net, double err, std::vector <ll> RealCom);
    void ApplyMultLacPro(NetMan & net, std::vector < std::shared_ptr <LAC> > pLacs, double backErr);
    void ApplyMultLac(NetMan & net, std::vector < std::shared_ptr <LAC> > pLacs, double backErr);   // wenhui
    double ApplyLacPro(NetMan & net, std::shared_ptr <LAC> pLac, double backErr);
    void ApplyLacCon(NetMan & net, std::shared_ptr <LAC> pLac, double backErr);
    void ExactSimpl(NetMan & net, ll round, bool fModifyfGenSub);
    std::vector <std::pair <ll, double>> GetVerTrun(NetMan & net, std::vector <ll> & TargNdoes, ll truncBit);
    void Eval(NetMan & net, const std::string & outpPath, double err, ll round);
    void Eval_app(NetMan & net, const std::string & outpPath, double err, ll round);
    void EvalPro(NetMan & net, const std::string & outpPath, double err, ll round, const std::string mark);
    std::vector < std::vector <ll> > TempApplyLacs(NetMan & net, std::vector < std::shared_ptr <LAC> > & lacs, LAC_TYPE lacType, bool isVerb);
    
    bool RunALS(const std::string & outpPath);
    double SimplByWinRewrite(NetMan & net, const std::string & outpPath, ll round, std::vector <ll> RealCom);
    bool ApplyRW(NetMan & net, std::shared_ptr <AppRW> pBestRW, double backArea, bool fPrint = false);
    void GenDB_singleOutput();
    void ObserveArea();
    void GenDB_appFunc();
    std::shared_ptr <AppRW> SelectBestRW(NetMan & net, SubCktMan & subcktMan, double backArea, double backErr);
    void MeasureCkt(const std::string & outpPath);
};
double CalcScore(double oldArea, double newArea, double oldDelay, double newDelay, double accArea, double accDelay, double deltaErr);