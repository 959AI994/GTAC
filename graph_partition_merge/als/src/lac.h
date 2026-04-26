#pragma once


#include "my_abc.h"
#include "header.h"
#include "simulator.h"
#include "espressoApi.h"


enum class LAC_TYPE {
    CONS, SASIMI, RAC, SUB_WIRE
};

enum class METR_TYPE{
    ER, MED, ME, MSE, SNR, MAXED, SELF, MRED, MHD
};      
// NMED(NMHD) is also supported, the same as MED(MHD)



static inline std::ostream & operator << (std::ostream & os, const LAC_TYPE lacType) {
    const std::string strs[4] = {"CONST", "SASIMI", "RAC", "SUB_WIRE"};
    os << strs[static_cast <ll> (lacType)];
    return os;
}


class LAC {
private:
    bigInt errBigInt;   // used currently (error increase sum for all patterns)
    bigFlt errBigFlt;
    double err;     // error increase which is divided by nFrame 
    ll targId;

public:
    explicit LAC() = default;
    explicit LAC(double _err, ll targ_node_id): err(_err), targId(targ_node_id) {}
    virtual ~LAC() = default;
    LAC(const LAC & oth_lac) = default;
    LAC(LAC &&) = default;
    LAC & operator = (const LAC & oth_lac) = default;
    LAC & operator = (LAC &&) = default;

    inline double GetErr() const {return err;}
    inline void SetErr(double _err) {err = _err;}
    inline bigInt GetErrPro() const {return errBigInt;}
    inline void SetErrPro(bigInt & err_big_int) {errBigInt = err_big_int;}
    inline ll GetTargId() const {return targId;}
    inline void SetTargId(ll targ_node_id) {targId = targ_node_id;}
    inline void Print(bool isNewLine) const {std::cout << "target node id = " << targId << ", " << "error = " << errBigInt; if (isNewLine) std::cout << std::endl;}
    inline void SetErrBigFlt(bigFlt err_) {errBigFlt = err_;}
    inline bigFlt GetErrBigFlt() const {return errBigFlt;}
};


class ConstLAC: public LAC {
private:
    bool isConst0;

public:
    explicit ConstLAC() = default;
    explicit ConstLAC(double _err, ll targ_node_id, bool is_const0): LAC(_err, targ_node_id), isConst0(is_const0) {}
    ~ConstLAC() = default;
    ConstLAC(const ConstLAC & oth_lac) = default;
    ConstLAC(ConstLAC && oth_lac) = default;
    ConstLAC & operator = (const ConstLAC & oth_lac) = default;
    ConstLAC & operator = (ConstLAC && oth_lac) = default;

    inline bool IsConst0() const {return isConst0;}
    inline void SetConst0(bool is_const0) {isConst0 = is_const0;}
    inline void Print(bool isNewLine = true) const {LAC::Print(false); std::cout << ", " << (isConst0? "const0": "const1"); if (isNewLine) std::cout << std::endl;}
};


class SasimiLAC: public LAC {
private:
    ll subId;
    bool isInv;

public:
    explicit SasimiLAC() = default;
    explicit SasimiLAC(double _err, ll targ_node_id, ll sub_node_id, bool is_inv): LAC(_err, targ_node_id), subId(sub_node_id), isInv(is_inv) {}
    ~SasimiLAC() = default;
    SasimiLAC(const SasimiLAC & oth_lac) = default;
    SasimiLAC(SasimiLAC && oth_lac) = default;
    SasimiLAC & operator = (const SasimiLAC & oth_lac) = default;
    SasimiLAC & operator = (SasimiLAC && oth_lac) = default;

    inline ll GetSubId() const {return subId;}
    inline bool GetIsInv() const {return isInv;}
    inline void Print(bool isNewLine = true) const {LAC::Print(false); std::cout << ", substitute node id = " << subId << ", " << (isInv? "inv": "buf"); if (isNewLine) std::cout << std::endl;}
};


class RacLAC: public LAC {
private:
    std::vector <ll> divs;
    std::string sop;

public:
    explicit RacLAC() = default;
    explicit RacLAC(double _err, ll targ_node_id, const std::vector <ll> & _divs, const std::string & _sop): LAC(_err, targ_node_id), divs(_divs), sop(_sop) {}
    ~RacLAC() = default;
    RacLAC(const RacLAC & oth_lac) = default;
    RacLAC(RacLAC && oth_lac) = default;
    RacLAC & operator = (const RacLAC & oth_lac) = default;
    RacLAC & operator = (RacLAC && oth_lac) = default;

    inline std::vector <ll> GetDivIds() const {return divs;}
    inline std::string GetSop() const {return sop;}
    inline void Print(bool isNewLine = true) const {LAC::Print(false); std::cout << ", divisors = "; PrintVect(divs, ",\n"); std::cout << sop; if (isNewLine) std::cout << std::endl;}
};


class SubWireLAC: public LAC {
private:
    ll subId;
    ll iFanin;
    bool isInv;

public:
    explicit SubWireLAC() = default;
    explicit SubWireLAC(double _err, ll targ_node_id, ll sub_node_id, ll i_fanin, bool is_inv): LAC(_err, targ_node_id), subId(sub_node_id), iFanin(i_fanin), isInv(is_inv) {}
    ~SubWireLAC() = default;
    SubWireLAC(const SubWireLAC & oth_lac) = default;
    SubWireLAC(SubWireLAC && oth_lac) = default;
    SubWireLAC & operator = (const SubWireLAC & oth_lac) = default;
    SubWireLAC & operator = (SubWireLAC && oth_lac) = default;

    inline ll GetSubId() const {return subId;}
    inline ll GetIFanin() const {return iFanin;}
    inline bool GetIsInv() const {return isInv;}
    inline void Print(bool isNewLine = true) const {LAC::Print(false); std::cout << ", substitute node id = " << subId << ", iFanin = " << iFanin << ", " << (isInv? "inv": "buf"); if (isNewLine) std::cout << std::endl;}
};


class LACMan {
private:
    // const ll maxHighAccNumb = 1024;
    const ll maxHighAccNumb = 20000;    // can be tuned
    std::vector < std::shared_ptr <LAC> > pLacs;    // use this √
    std::vector < std::shared_ptr <LAC> > candLacs;     // do not use this

public:
    explicit LACMan() = default;
    ~LACMan() = default;
    LACMan(const LACMan &) = delete;
    LACMan(LACMan &&) = delete;
    LACMan & operator = (const LACMan &) = delete;
    LACMan & operator = (LACMan &&) = delete;
    void GenConstLACs(NetMan & net, std::vector <ll> & targIds);
    void GenConstLACs_ForUpdate(NetMan & net);
    void GenSasimiLACsAll(NetMan & net, std::vector <ll> & targIds);
    void GenSasimiLACsNew(NetMan & net, std::vector <ll> & targIds);
    void GenRacLACsNew(NetMan & net, std::vector <ll> & targIds, unsigned seed);
    void GenSubWireLACs(NetMan & net, std::vector <ll> & targIds);
    void Filt(double perc);
    void FiltPro(double perc, NetMan & net);
    std::shared_ptr <LAC> GetBestLac() const;
    std::shared_ptr <LAC> GetBestLacPro() const;
    std::vector < std::shared_ptr <LAC> > GetMultBestLac() const;
    // std::vector <std::shared_ptr <LAC>> GetNegErrLac() const;
    std::vector <ll> GetDivs(abc::Abc_Obj_t * pNode, ll nLevDivMax);
    std::string BuildFuncWithEspresso(Simulator & smlt, abc::Abc_Obj_t * pPivot, const std::vector <ll> & faninIds);
    void GenCandLacs();
    void GenCandLacs(const std::vector <ll> & critGraph);

    inline ll GetLacNum() const {return static_cast <ll> (pLacs.size());}
    inline std::shared_ptr <LAC> GetLac(ll i) const {return pLacs[i];}
    inline ll GetCandLacSize() const {return candLacs.size();}
    inline std::shared_ptr <LAC> GetCandLac(ll id) const {return candLacs[id];}

    void PrintLACsErr();
    std::set <ll> GetScand(ll nCand, METR_TYPE metrType, NetMan & net, ll nFrame);
    std::shared_ptr <LAC> GetLacWithSmallestErr(METR_TYPE metrType);
    void SortLacs(METR_TYPE metrType);
    std::shared_ptr <LAC> GetLac(ll i);
    void CleanLacs();
    bool CheckSasimiLev(NetMan & net);
};

enum class SNG_OBJ_TYPE {
    SNG_NONE, SNG_NODE, SNG_LO, SNG_LI
};

class SNGNode {
private:
    abc::Abc_Obj_t * pNode;
    ll SNGId;   // start from 1
    SNG_OBJ_TYPE Type;
    std::vector <ll> vTotalFanouts;
    std::vector <ll> vTotalFanins;
    std::vector <ll> vDirFanouts;
    std::vector <ll> vDirFanins;
    std::vector <ll> vIndFanouts;
    std::vector <ll> vIndFanins;
    bool fMarkA;
    bool fMarkB;
    ll travId;
    ll nMffc;   // at least 1 (itself)

public:
    explicit SNGNode() : pNode(nullptr), SNGId(0), Type(SNG_OBJ_TYPE::SNG_NONE), fMarkA(0), travId(0), nMffc(0) {
        // vector is automatically initialized as empty.
    }
    explicit SNGNode(abc::Abc_Obj_t * pNode, ll id) : pNode(pNode), SNGId(id), Type(SNG_OBJ_TYPE::SNG_NODE), fMarkA(0), travId(0), nMffc(0) {
        // vector is automatically initialized as empty.
    }
    void AddDirFanin(ll SNGId) {vDirFanins.push_back(SNGId); vTotalFanins.push_back(SNGId);}
    void AddDirFanout(ll SNGId) {vDirFanouts.push_back(SNGId); vTotalFanouts.push_back(SNGId);}
    void AddIndFanin(ll SNGId) {vIndFanins.push_back(SNGId); vTotalFanins.push_back(SNGId);}
    void AddIndFanout(ll SNGId) {vIndFanouts.push_back(SNGId); vTotalFanouts.push_back(SNGId);}
    bool FindIdInTotalFanouts(ll id);
    bool FindIdInTotalFanins(ll id);

    inline ll GetSNGId() const {return SNGId;}
    inline ll GetNetId() const {return abc::Abc_ObjId(pNode);}
    inline ll GetTotalFaninNum() const {return vTotalFanins.size();}
    inline ll GetTotalFanoutNum() const {return vTotalFanouts.size();}
    inline void SetType(SNG_OBJ_TYPE newType) {Type = newType;}
    inline SNG_OBJ_TYPE GetType() {return Type;}
    inline bool GetfMarkA() const {return fMarkA;}
    inline void SetfMarkA() {fMarkA = true;}
    inline void ResetfMarkA() {fMarkA = false;}
    inline bool GetfMarkB() const {return fMarkB;}
    inline void SetfMarkB() {fMarkB = true;}
    inline void ResetfMarkB() {fMarkB = false;}
    inline ll GetTravId() const {return travId;}
    inline void SetTravId(ll newTravId) {travId = newTravId;}
    inline bool IsTravIdCurrent(ll currTravId) {return (travId == currTravId);}
    inline ll GetFanin(ll i) {return vTotalFanins[i];}
    inline ll GetFanout(ll i) {return vTotalFanouts[i];}
    inline abc::Abc_Obj_t * GetpNode() {return pNode;}
    void SetnMffc(NetMan & net);
    inline ll GetnMffc() const {return nMffc;}
    // void FindIndFanins();
};

class SNGMan {
private:
    std::vector <std::shared_ptr <SNGNode>> vSubNodes;   // vector index starts from 0, but id starts from 1. 
    std::vector <std::shared_ptr <SNGNode>> vLos;   // local outputs
    std::vector <std::shared_ptr <SNGNode>> vLis;   // local inputs
    std::vector <double> errInc;    // LAC with smallest error increase on each node
    ll nNodes;
    ll currTravId;

public:
    explicit SNGMan() 
        : nNodes(0), currTravId(0) {
        // vector is automatically initialized as empty.
    }
    void Clear();
    void CreateNode(abc::Abc_Obj_t * pNode);
    void CreateNode(std::shared_ptr <SNGNode> pSNGNode);
    ll IncrementnNodes() {++nNodes; return nNodes;}
    std::shared_ptr <SNGNode> GetNode(ll id) const;
    inline ll GetnNodes() const {return nNodes;}
    void AddLocalInput(std::shared_ptr <SNGNode> pLi) {vLis.push_back(pLi);}
    void AddLocalOutput(std::shared_ptr <SNGNode> pLo) {vLos.push_back(pLo);}
    void ClearGraph();
    void UpdateErrInc(std::unordered_map <ll, std::shared_ptr <LAC>> & LacPerSubNode);
    double GenNearDisCut(NetMan & net, std::vector <ll> & nearDisCut, double errUppBound, double backErr, METR_TYPE metrType, double jointTh, ll & nRoundLowGain);
    inline ll GetCurrTravId() const {return currTravId;}
    inline void IncrementCurrTravId() {++currTravId;}
    void MarkTFO_rec(ll id);
    void MarkTFI_rec(ll id);
    void CleanfMarkA();
    void CleanfMarkB();
    inline ll GetLiNum() {return vLis.size();}
    inline ll GetLoNum() {return vLos.size();}
    inline ll GetNodeNum() {return vSubNodes.size();}
    void UpdatenMffc(NetMan & net);
    ll FindBestSingleLAC(NetMan & net, double errUppBound, double backErr, METR_TYPE metrType);
};

double CalcNormErr(METR_TYPE metrType, double backErr, double errInc);
double CalcNormErr2(METR_TYPE metrType, double backErr, double errInc);

std::pair<double, ll> SynthFunction(ll tableValue, ll nVars);
double SynthFunction_MultiOut(std::vector<ll> tableValues, ll nVars);