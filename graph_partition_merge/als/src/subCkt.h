#pragma once

#include "header.h"
#include "my_abc.h"
#include "simulator.h"
#include "error.h"
#include "database.h"

class SubCkt {
private:
    std::vector <ll> vLO;
    std::vector <ll> vLI;
    double area;
    ll nodeNum;
    std::vector < std::vector <ll> > vDiv;
    double errIncSum;
    double minErr;
    ll rank1;   // use errIncSum
    ll rank2;   // use minErr

public:
    SubCkt(double area_, ll nodeNum_, const std::vector<ll>& vLI_, const std::vector<ll>& vLO_)
        : area(area_), nodeNum(nodeNum_), vLI(vLI_), vLO(vLO_) {
            errIncSum = 0.0;
            minErr = 0.0;
            rank1 = -1;
            rank2 = -1;
        }
    
    void SetArea(double areaNew) {area = areaNew;}
    double GetArea() const {return area;}
    ll GetNodeNum() {return nodeNum;}
    const std::vector<ll> GetvLO() const { return vLO; }
    const std::vector<ll> GetvLI() const { return vLI; }
    ll GetLInum() const {return vLI.size();}
    ll GenDiv(NetMan & net, ll nlevLim);
    const std::vector<std::vector<ll>> GetvDiv() const { return vDiv; }
    void SetErrIncSum(double errIncSumNew) {errIncSum = errIncSumNew;}
    double GetErrIncSum() const {return errIncSum;}
    void SetMinErr(double minErrNew) {minErr = minErrNew;}
    double GetMinErr() const {return minErr;}
    void SetRank1(ll rankNew) {rank1 = rankNew;}
    ll GetRank1() const {return rank1;}
    void SetRank2(ll rankNew) {rank2 = rankNew;}
    ll GetRank2() const {return rank2;}
};

struct PattValue {
    ll n0;  // num of 0s
    ll n1;  // num of 1s
    ll d;   // num of don't-cares
};

struct FlatTtMark {
    ll LoId;
    ll divPattId;
    ll mark;
};

struct Subset {
    ll sum;
    std::vector <ll> indices;
    ll lastIndex;

    bool operator>(const Subset& other) const {
        return sum > other.sum;
    }
};

class AppRW {
private:
    std::vector <ll> vLO;
    std::vector <ll> vDiv;
    std::vector <ll> appFunc;
    double reArea;
    double oriArea;
    double error;
    int fRelation;
    ll vLoId;
    ll hd;
    double score;
    std::vector <ll> f0;
    double reDelay;

public:
    AppRW(const std::vector<ll>& vLO_,
          const std::vector<ll>& vDiv_,
          const std::vector<ll>& appFunc_,
          double reArea_,
          double oriArea_,
          double error_,
          int fRelation_,
          ll vLoId_,
          ll hd_,
          const std::vector<ll>& f0_)
        : vLO(vLO_), vDiv(vDiv_), appFunc(appFunc_), reArea(reArea_), oriArea(oriArea_), error(error_), fRelation(fRelation_), vLoId(vLoId_), hd(hd_), f0(f0_) {
            score = 0; 
            reDelay = 0;}
    inline double GetReArea() const {return reArea;}
    inline double GetOriArea() const {return oriArea;}
    inline double GetError() const {return error;}
    std::vector <ll> GetFuncs() const {return appFunc;}
    std::vector <ll> GetvLO() const {return vLO;}
    std::vector <ll> GetvDiv() const {return vDiv;}
    inline ll GetnVars() const {return vDiv.size();}
    inline ll GetnLOs() const {return vLO.size();}
    void Print();
    void PrintOriSubNtk(NetMan & net);
    inline ll GetvLoId() const {return vLoId;}
    inline void SetError(double errorNew) {error = errorNew;}
    inline ll GetHd() const {return hd;}
    inline void SetScore(double scoreNew) {score = scoreNew;}
    inline double GetScore() {return score;}
    inline void SetReArea(double reAreaNew) {reArea = reAreaNew;}
    inline void SetReDelay(double reDelayNew) {reDelay = reDelayNew;}
    inline double GetReDelay() {return reDelay;}
};

class DivInfo {
private:
    ll divId;
    ll rank;    // start from 0
    std::vector < std::vector <ll> > hamMarks;
    ll nIniFlipBits;
    std::vector <ll> vIniFeasibleTt;

public:
    DivInfo(const ll divId_, const std::vector < std::vector <ll> > hamMarks_, const ll nIniFlipBits_, const std::vector <ll> vIniFeasibleTt_) : divId(divId_), hamMarks(hamMarks_), nIniFlipBits(nIniFlipBits_), vIniFeasibleTt(vIniFeasibleTt_) {
        rank = -1;
    } 
    inline void SetRank(ll rankNew) {rank = rankNew;}
    inline ll GetHd0() const {return nIniFlipBits;}
    inline ll GetDivId() const {return divId;}
    inline const std::vector < std::vector <ll> > & GetHamMarks() const {return hamMarks;}
    inline const std::vector <ll> & GetvIniFeasibleTt() const {return vIniFeasibleTt;}
    inline std::vector <ll> GetF0() {return vIniFeasibleTt;}
};

class SubCktMan {
private:
    NetMan & net;
    std::vector < std::shared_ptr <SubCkt> > pSubCkts2;     // #LO = 2
    std::vector < std::shared_ptr <SubCkt> > pSubCkts3;     // #LO = 3
    std::vector <int> vLO2Relation;     // 0: no TFI/TFO relationship, 1: pObj1 is a TFI of pObj2

    std::vector < std::shared_ptr <SubCkt> > pSubCkts2Trivial;     // #LO = 2
    std::vector < std::shared_ptr <SubCkt> > pSubCkts3Trivial;     // #LO = 3
    

    // CPM
    std::vector < std::vector < boost::dynamic_bitset <ull> > > bdPo2Nodes11; // bdPo2Node[poId][nodeId]
    std::vector < std::vector < boost::dynamic_bitset <ull> > > bdPo2Nodes10;
    std::vector < std::vector < boost::dynamic_bitset <ull> > > bdPo2Nodes101;
    std::vector < std::vector < boost::dynamic_bitset <ull> > > bdPo2Nodes110;
    std::vector < std::vector < boost::dynamic_bitset <ull> > > bdPo2Nodes011;
    std::vector < std::vector < boost::dynamic_bitset <ull> > > bdPo2Nodes111;

    std::vector < std::shared_ptr <AppRW> > pAppRWs;
    std::mutex appRwMutex;
    
    METR_TYPE metrType;
    ll nOutput;
    ll nFrame;
    bool isSign;
    double errUppBound;
    double backErr;

    ll maxAppRWNum;     // for one specific LO set
    std::vector < std::vector < std::shared_ptr <AppRW> > > pAppRWs2;   // [vLoId][appRW's Id]
    std::vector < std::vector < std::shared_ptr <AppRW> > > pAppRWs3;
    std::vector < std::set <ll> > hdRank2;    // [vLoId][appRW's rank number (<=maxAppRWNum)], the value is hamming distance
    std::vector < std::set <ll> > hdRank3;

    std::vector < std::vector <DivInfo> > divInfo2;    // [vLoId][vDiv's Id]. delete after using
    std::vector < std::vector <DivInfo> > divInfo3;
    std::mutex mtx1;
    std::mutex mtx2;    

public:
    SubCktMan(NetMan& net_, METR_TYPE metrType_, ll nOutput_, ll nFrame_, bool isSign_, double errUppBound_, double backErr_, ll maxAppRWNum_)
        : net(net_), metrType(metrType_), nOutput(nOutput_), nFrame(nFrame_), isSign(isSign_), errUppBound(errUppBound_), backErr(backErr_), maxAppRWNum(maxAppRWNum_) {
        // Other vectors are default-initialized as empty
    }
    void GenSubCkts(const std::vector <ll> & Scand);
    void Print();
    inline ll GetSubCktNum() {return (pSubCkts2.size() + pSubCkts3.size());}
    inline ll GetSubCktNum2() {return pSubCkts2.size();}
    inline ll GetSubCktNum3() {return pSubCkts3.size();}
    inline ll GetSubCktNum2T() {return pSubCkts2Trivial.size();}
    inline ll GetSubCktNum3T() {return pSubCkts3Trivial.size();}
    void CalcBD(Simulator & appSmlt, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef_, const std::vector<boost::dynamic_bitset<ull>>& poMarks_, const std::vector <ll> & topoIds);
    void GenDivs();
    void GenAllAppRWs(Simulator & appSmlt, Simulator & accSmlt, Database & db, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef, ll nThread);
    void GenAllAppRWsPro(Simulator & appSmlt, Simulator & accSmlt, Database & db, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef, ll nThread);
    void GenAppRW(std::vector <ll> vDiv, std::vector <ll> vLO, ll vLoId, Simulator & appSmlt, Simulator & accSmlt, Database & db, double accArea, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef);
    void GenAppRWPro(std::vector <ll> vDiv, std::vector <ll> vLO, ll vLoId, Simulator & appSmlt, Simulator & accSmlt, Database & db, double accArea, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef, ll divId);
    void PreGenAppRW(std::vector <ll> vDiv, std::vector <ll> vLO, ll vLoId, Simulator & appSmlt, ll divId, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef);
    int CheckRW(std::vector <ll> vTt, std::vector <ll> vDiv, std::vector <ll> vLO, ll vLoId, Simulator & appSmlt, Simulator & accSmlt, Database & db, double accArea, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef);
    int CheckRW_debug(std::vector <ll> vTt, std::vector <ll> vDiv, std::vector <ll> vLO, ll vLoId, Simulator & appSmlt, Simulator & accSmlt, Database & db, double accArea, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef);
    double CalcRwErr(std::vector <ll> vTt, std::vector <ll> vDiv, std::vector <ll> vLO, ll vLoId, Simulator & appSmlt, Simulator & accSmlt, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef, ll nFrameHere = 0);
    void PrintAppRWs();
    std::shared_ptr <AppRW> SelectBestAppRW(double backErr);
    inline ll GetAppRwNum() {return pAppRWs.size();}
    inline std::shared_ptr <AppRW> GetAppRW(ll i) {return pAppRWs[i];}  // don't check
    void CleanForTrivialCase();
    void SortAppRWs();
    void SortAppRWsByErr();
    ll GetSubCktRank1(ll nLo, ll vLoId) const;
    ll GetSubCktRank2(ll nLo, ll vLoId) const;
    void BatchErr(Simulator & appSmlt, Simulator & accSmlt, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef, ll nThread);
    ll GetHdTh(ll nLo, ll vLoId);   // threshold
};

std::vector<ll> getUniqueSorted(std::vector<ll> v);
std::vector<ll> mergeSortedDivs(const std::vector<ll>& div1, const std::vector<ll>& div2);
void simplifyDivs(std::vector<std::vector<ll>>& vDiv);