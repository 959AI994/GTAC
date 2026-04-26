#pragma once


#include "simulator.h"
#include "my_abc.h"
#include "lac.h"


// enum class METR_TYPE{
//     ER, MED, ME, MSE, SNR, MAXED, SELF, MRED
// };      // NMED is also supported, the same as MED


enum class DISTR_TYPE {
    UNIF, ENUM, MIX, SELF
};


static inline std::ostream & operator << (std::ostream & os, const METR_TYPE metrType) {
    const std::string strs[9] = {"ER", "MED", "ME", "MSE", "SNR", "MAXED", "SELF", "MRED", "MHD"};
    os << strs[static_cast <ll> (metrType)];
    return os;
}


static inline std::ostream & operator << (std::ostream & os, const DISTR_TYPE distrType) {
    const std::string strs[4] = {"UNIF", "ENUM", "MIX", "SELF"};
    os << strs[static_cast <ll> (distrType)];
    return os;
}


class ErrManPro {
private:
    NetMan& net0;
    NetMan& net1;
    std::shared_ptr <NetMan> pProc;
    std::shared_ptr <NetMan> pMit;
    std::shared_ptr <Simulator> pMitSmlt;
    bool isSign;
    unsigned seed;
    int nNetPI;
    int nNetPo;
    ll nFrame;
    METR_TYPE metrType;
    DISTR_TYPE distrType;

public:
    ErrManPro(NetMan& _net0, NetMan& _net1, bool is_sign, unsigned _seed, ll n_frame, METR_TYPE metr_type, DISTR_TYPE distr_type);
    ~ErrManPro() = default;
    ErrManPro(const ErrManPro &) = delete;
    ErrManPro(ErrManPro &&) = delete;
    ErrManPro & operator = (const ErrManPro &) = delete;
    ErrManPro & operator = (ErrManPro &&) = delete;
    void InitMit();
    void CreateBehLevMit(const std::string& fileName);
    double CalcErr();
};


class ErrMan {
private:
    NetMan & net0;
    NetMan & net1;
    std::shared_ptr <Simulator> pSmlt0;
    std::shared_ptr <Simulator> pSmlt1;
    unsigned seed;
    ll nFrame;
    ll nOutput;
    DISTR_TYPE distrType;
    // std::string selfDefDistr;

public:
    ErrMan(NetMan & netMan0, NetMan & netMan1, unsigned _seed, ll n_frame, ll nOutput, DISTR_TYPE distr_type);
    ~ErrMan() = default;
    ErrMan(const ErrMan &) = delete;
    ErrMan(ErrMan &&) = delete;
    ErrMan & operator = (const ErrMan &) = delete;
    ErrMan & operator = (ErrMan &&) = delete;

    void InitForStatErr();
    double CalcErrRate(bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId);
    double CalcMeanErrDist(bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId);
    double CalcMeanErr(bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId);
    double CalcMeanSquareErr(bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId);
    double CalcSigNoiseRat(bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId);
    double CalcMaxErrDist(bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId);
    double CalcMeanRelErrDist(bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId);
    double CalcMeanHamDist(bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId);
    // double CalcSelfDefErr(bool isSign, const std::string & selfDefMetr);
    // ull CalcMaxErrDist(bool isSign);
    double CalcMeanSquareErr_forDebug(bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId);
    ull GET_MEM(abc::Abc_Ntk_t * pNtk1, abc::Abc_Ntk_t * pNtk2);
    ull NtkMiterComp(abc::Abc_Ntk_t * pNtk1, abc::Abc_Ntk_t * pNtk2);
    ull NtkMiterFinalize( abc::Abc_Ntk_t * pNtk1, abc::Abc_Ntk_t * pNtk2, abc::Abc_Ntk_t * pNtkMiter, ll fComb, ll nPartSize, ll fImplic, ll fMulti );
    abc::Abc_Obj_t ** X_subtract_Y_abs(abc::Abc_Ntk_t * pNtk, abc::Abc_Obj_t * X[], abc::Abc_Obj_t * Y[], ll n);
    ull GETMEM(abc::Abc_Ntk_t * pNtk, abc::Abc_Obj_t *R[], ll n);
    abc::Abc_Obj_t * X_lt_Y(abc::Abc_Ntk_t * pNtk, abc::Abc_Obj_t * X[], abc::Abc_Obj_t * Y[], ll n);
    bool SATSolver(abc::Abc_Ntk_t * pNtk);
};


double CalcErrPro(NetMan& net0, NetMan& net1, bool isSign, unsigned seed, ll nFrame, METR_TYPE metrType, DISTR_TYPE distrType);
double CalcErr(NetMan & netMan0, NetMan & netMan1, bool isSign, unsigned seed, ll nFrame, ll nOutput, METR_TYPE metrType, DISTR_TYPE distrType, std::vector <ll> RealCom, ll cutId);
double CalcErr_forDebug(NetMan & netMan0, NetMan & netMan1, bool isSign, unsigned seed, ll nFrame, ll nOutput, METR_TYPE metrType, DISTR_TYPE distrType, std::vector <ll> RealCom, ll cutId);
double GetMSEFromSNR(NetMan & net, bool isSign, unsigned seed, ll nFrame, DISTR_TYPE distrType, double snr, ll nOutput);
// bool CheckMaxErrDist(NetMan & netMan0, NetMan & netMan1, ll maxErrDist);


class VECBEEMan {
private:
    bool isSign;
    unsigned seed;
    ll nFrame;
    ll nOutput;
    METR_TYPE metrType;
    LAC_TYPE lacType;
    DISTR_TYPE distrType;
    const ll nThread;
    std::vector < std::vector < boost::dynamic_bitset <ull> > > bdPo2Nodes; // bdPo2Node[poId][nodeId]
    std::vector < std::vector < boost::dynamic_bitset <ull> > > bdCut2Nodes; // bdCut2Node[nodeId][cutId]
    std::vector < std::list <abc::Abc_Obj_t *> > disjCuts;
    std::vector < std::vector <abc::Abc_Obj_t *> > cutNtks;
    std::vector < boost::dynamic_bitset <ull> > poMarks;
    std::vector <ll> topoIds;

public:
    VECBEEMan() = default;
    VECBEEMan(bool is_sign, unsigned _seed, ll n_frame, ll n_output, METR_TYPE metr_type, LAC_TYPE lac_type, DISTR_TYPE distr_type, ll n_thread):
        isSign(is_sign), seed(_seed), nFrame(n_frame), nOutput(n_output), metrType(metr_type), lacType(lac_type), distrType(distr_type), nThread(n_thread) {}
    ~VECBEEMan() = default;
    VECBEEMan(const VECBEEMan &) = delete;
    VECBEEMan(VECBEEMan &&) = delete;
    VECBEEMan & operator = (const VECBEEMan &) = delete;
    VECBEEMan & operator = (VECBEEMan &&) = delete;

    void BatchErrEstPro(NetMan & accNet, NetMan & appNet, LACMan & lacMan, const bigInt & upperBound, bool useAppDisjCut, ll nOutput, std::vector <ll> RealCom);
    void FindDisjCut(NetMan & net, std::vector <abc::Abc_Obj_t *> & topoNodes);
    void FindAppDisjCut(NetMan & net);
    void FindDisjCutOfNode(abc::Abc_Obj_t * pObj, std::list <abc::Abc_Obj_t *> & disjCut);
    // void ExpandCut(abc::Abc_Obj_t * pObj, std::list <abc::Abc_Obj_t *> & disjCut);
    abc::Abc_Obj_t * ExpandWhich(std::list <abc::Abc_Obj_t *> & disjCut);
    void CalcBoolDiffCut2Node(Simulator & appSmlt, std::vector <abc::Abc_Obj_t *> & topoNodes);
    void CalcBoolDiffPo2Node(Simulator & appSmlt, std::vector <abc::Abc_Obj_t *> & topoNodes);
    void CalcBoolDiffPo2NodePlus(Simulator & appSmlt, std::vector <abc::Abc_Obj_t *> & topoNodes);
    void CalcLACErrsPlus(Simulator & accSmlt, Simulator & appSmlt, LACMan & lacMan, const bigInt & upperBound, ll nOutput, bool useAppDisjCut, std::vector <ll> RealCom);

    void BatchErrEstPro_GetLacPerNetNode(NetMan & accNet, NetMan & appNet, LACMan & lacMan, const bigInt & uppBound, bool useAppDisjCut, ll nOutput, std::vector <ll> RealCom, std::unordered_map <ll, std::shared_ptr <LAC>> & LacPerNode, bool fFilt);
    void BatchErrEstPro_GetLacPerSubNode(NetMan & accNet, NetMan & appNet, LACMan & lacMan, const bigInt & uppBound, bool useAppDisjCut, ll nOutput, std::vector <ll> RealCom, std::vector <std::shared_ptr <LAC>> & LacPerNode);
    void CalcLACErrsPlus_GetLacPerNode(Simulator & accSmlt, Simulator & appSmlt, LACMan & lacMan, const bigInt & upperBound, ll nOutput, bool useAppDisjCut, std::vector <ll> RealCom, std::unordered_map <ll, std::shared_ptr <LAC>> & LacPerNode, ll idMaxPlus1, bool fFilt, NetMan & net);

    void BatchErrEst(NetMan & appNet, Simulator & accSmlt, Simulator & appSmlt, LACMan & lacMan, const bigInt & upperBound, ll nOutput, std::vector <ll> RealCom, ll nCand);
    void CalcLACErrs(Simulator & accSmlt, Simulator & appSmlt, LACMan & lacMan, const bigInt & upperBound, ll nOutput, std::vector <ll> RealCom, ll nCand);

    const std::vector<std::vector<boost::dynamic_bitset<ull>>>& getBdPo2Nodes() const {return bdPo2Nodes;}
    const std::vector<boost::dynamic_bitset<ull>>& getPoMarks() const {return poMarks;}
    const std::vector<ll>& getTopoIds() const {return topoIds;}
    void CleanForNewLacCalc();
};
void ExpandCut(abc::Abc_Obj_t * pObj, std::list <abc::Abc_Obj_t *> & disjCut);

class VECBEEManPro {
private:
    const std::vector < std::vector <ll> > & LOs2;
    const std::vector < std::vector <ll> > & LOs3;
    const std::vector <ll> & topoIds;

    const std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2NodesRef;  // single node's bd
    const std::vector < boost::dynamic_bitset <ull> > & poMarks;

    std::vector < std::list <abc::Abc_Obj_t *> > djCuts2;
    std::vector < std::vector <abc::Abc_Obj_t *> > cutNtks2;
    std::vector < std::vector < boost::dynamic_bitset <ull> > > bdCut2Nodes11;
    std::vector < std::vector < boost::dynamic_bitset <ull> > > bdCut2Nodes10;
    std::vector < std::list <abc::Abc_Obj_t *> > djCuts3;
    std::vector < std::vector <abc::Abc_Obj_t *> > cutNtks3;
    std::vector < std::vector < boost::dynamic_bitset <ull> > > bdCut2Nodes101;
    std::vector < std::vector < boost::dynamic_bitset <ull> > > bdCut2Nodes110;
    std::vector < std::vector < boost::dynamic_bitset <ull> > > bdCut2Nodes011;
    std::vector < std::vector < boost::dynamic_bitset <ull> > > bdCut2Nodes111;

public:
    VECBEEManPro(const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef_,
                 const std::vector<boost::dynamic_bitset<ull>>& poMarks_,
                 const std::vector<std::vector<ll>>& LOs2_,
                 const std::vector<std::vector<ll>>& LOs3_,
                 const std::vector<ll>& topoIds_)
        : bdPo2NodesRef(bdPo2NodesRef_), poMarks(poMarks_), LOs2(LOs2_), LOs3(LOs3_), topoIds(topoIds_) {
        // Other vectors are default-initialized as empty
    }
    void BuildCutNtks(NetMan & net);
    void FindDisjointCutofMultNodes(NetMan & net, std::vector<ll> objIds, std::list <abc::Abc_Obj_t *> & djCut);
    abc::Abc_Obj_t * ExpandWhich(std::list <abc::Abc_Obj_t *> & disjCut);
    void CalcBdCut2Node(Simulator & appSmlt, std::vector <int> & vLO2Relation);
    void CalcBdPo2Node(Simulator & appSmlt);
    void CalcPoBd2(ll PoId, ll LoId, boost::dynamic_bitset <ull> & bdPo2Node11, boost::dynamic_bitset <ull> & bdPo2Node10);
    void CalcPoBd3(ll PoId, ll LoId, boost::dynamic_bitset <ull> & bdPo2Node101, boost::dynamic_bitset <ull> & bdPo2Node110, boost::dynamic_bitset <ull> & bdPo2Node011, boost::dynamic_bitset <ull> & bdPo2Node111);
};

double GetErrFromPoValue(const std::vector < boost::dynamic_bitset <ull> > & accDat, const std::vector < boost::dynamic_bitset <ull> > & appDat, bool isSign, ll nOutput, METR_TYPE metrType, bool fDebug, double errUppBound);
double GetErrRate(const std::vector < boost::dynamic_bitset <ull> > & accDat, const std::vector < boost::dynamic_bitset <ull> > & appDat, bool isSign, ll nOutput, double errUppBound);
double GetMeanErrDist(const std::vector < boost::dynamic_bitset <ull> > & accDat, const std::vector < boost::dynamic_bitset <ull> > & appDat, bool isSign, ll nOutput, double errUppBound);
double GetMeanSquareErr(const std::vector < boost::dynamic_bitset <ull> > & accDat, const std::vector < boost::dynamic_bitset <ull> > & appDat, bool isSign, ll nOutput, bool fDebug, double errUppBound);
double GetMeanRelErrDist(const std::vector < boost::dynamic_bitset <ull> > & accDat, const std::vector < boost::dynamic_bitset <ull> > & appDat, bool isSign, ll nOutput, double errUppBound);
double GetMeanHamDist(const std::vector < boost::dynamic_bitset <ull> > & accDat, const std::vector < boost::dynamic_bitset <ull> > & appDat, bool isSign, ll nOutput, double errUppBound);

boost::dynamic_bitset<ull> bigIntToBin(const bigInt& val, ll nPo, bool isSign);