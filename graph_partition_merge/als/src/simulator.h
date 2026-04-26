#pragma once


#include "header.h"
// #include "exprtkWrap.h"
#include "my_abc.h"


class Simulator: public NetMan {
private:
    unsigned seed;
    ll nFrame;
    std::vector < boost::dynamic_bitset <ull> > dat;
    std::vector < boost::dynamic_bitset <ull> > tempDat;

public:
    explicit Simulator(NetMan & net_man, unsigned _seed, ll n_frame);
    ~Simulator() = default;
    Simulator(const Simulator &) = delete;
    Simulator(Simulator &&) = delete;
    Simulator & operator = (const Simulator &) = delete;
    Simulator & operator = (Simulator &&) = delete;

    void InpUnif();
    void InpUnifFast();
    void InpEnum();
    void InpMix();
    void InpSelf(const std::string & fileName);
    void Sim();
    void UpdAigNode(abc::Abc_Obj_t * pObj);
    void UpdSopNode(abc::Abc_Obj_t * pObj);
    void UpdGateNode(abc::Abc_Obj_t * pObj);
    void UpdSop(abc::Abc_Obj_t * pObj, char * pSop);
    bigInt GetInp(ll iPatt, ll lsb, ll msb, bool isSign) const;
    void PrintInpStream(ll iPatt, bool isRev) const;
    bigInt GetOutp(ll iPatt) const;
    bigInt GetOutpPro(ll iPatt, bool isSign) const;
    std::vector<bigInt> GetOutpPro_complex(ll iPatt, bool isSign, ll nOutput) const;
    bigInt GetTempOutpPro(ll iPatt, bool isSign) const;
    ll GetOutpFast(ll iPatt, bool isSign) const;
    ll GetTempOutpFast(ll iPatt, bool isSign) const;
    void PrintOutpStream(ll iPatt) const;
    double GetSignalProb(ll objId) const;
    void PrintSignalProb() const;
    bool IsPIOSame(const Simulator & oth_smlt) const;
    double GetErrRate(const Simulator & oth_smlt, bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId, bool isCheck = false) const;
    double GetMeanErrDist(const Simulator & oth_smlt, bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId, bool isCheck = false) const;
    double GetMeanErr(const Simulator & oth_smlt, bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId, bool isCheck = false) const;
    double GetMeanSquareErr(const Simulator & oth_smlt, bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId, bool isCheck = false) const;
    double GetMeanSquareErr_forDebug(const Simulator & oth_smlt, bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId, bool isCheck = false) const;
    double GetSigNoiseRat(const Simulator & oth_smlt, bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId, bool isCheck = false) const;
    double GetMaxErrDist(const Simulator & oth_smlt, bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId, bool isCheck = false) const;
    double GetMeanRelErrDist(const Simulator & oth_smlt, bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId, bool isCheck = false) const;
    double GetMeanHamDist(Simulator & oth_smlt, bool isSign, ll nOutput, std::vector <ll> RealCom, ll cutId, bool isCheck = false);
    // double GetSelfDefErr(const Simulator & oth_smlt, bool isSign, const std::string & selfDefMetr, bool isCheck = false) const;
    void CalcLocBoolDiff(abc::Abc_Obj_t * pObj, std::list <abc::Abc_Obj_t *> & disjCut, std::vector <abc::Abc_Obj_t *> & cutNtk, std::vector < boost::dynamic_bitset <ull> > & bdCut2Node);
    void CalcLocPartDiff(abc::Abc_Obj_t * pObj, std::list <abc::Abc_Obj_t *> & disjCut, std::vector <abc::Abc_Obj_t *> & cutNtk, std::vector < std::vector <int8_t> > & pdCut2Node);
    void UpdSopNodeForBoolAndPartDiff(abc::Abc_Obj_t * pObj);
    void UpdGateNodeForBoolAndPartDiff(abc::Abc_Obj_t * pObj);
    void UpdSopForBoolAndPartDiff(abc::Abc_Obj_t * pObj, char * pSop);

    inline ll GetFrameNumb() {return nFrame;}
    inline boost::dynamic_bitset <ull> * GetDat(abc::Abc_Obj_t * pObj) {return &dat[GetId(pObj)];}
    inline boost::dynamic_bitset <ull> * GetDat(ll id) {return &dat[id];}
    inline bool GetDat(abc::Abc_Obj_t * pObj, ll frameId) {return dat[GetId(pObj)][frameId];}
    inline bool GetDat(ll id, ll frameId) {return dat[id][frameId];}
    inline boost::dynamic_bitset <ull> * GetTempDat(abc::Abc_Obj_t * pObj) {return &tempDat[GetId(pObj)];}
    inline boost::dynamic_bitset <ull> * GetTempDat(ll id) {return &tempDat[id];}
    inline void SetTempDat(ll id, boost::dynamic_bitset <ull> & newDat) {tempDat[id] = newDat;}
    inline void SetTempDat(ll id, boost::dynamic_bitset <ull> && newDat) {tempDat[id] = newDat;}
    inline int CountNumbOfOnes(int id) {return dat[id].count();}

    int CalcLocBd2(ll objId1, ll objId2, std::list <abc::Abc_Obj_t *> & disjCut, std::vector <abc::Abc_Obj_t *> & cutNtk, std::vector <boost::dynamic_bitset<ull>> & bdCut2Node11, std::vector <boost::dynamic_bitset<ull>> & bdCut2Node10);
    void CalcLocBd3(ll objId1, ll objId2, ll objId3, std::list <abc::Abc_Obj_t *> & disjCut, std::vector <abc::Abc_Obj_t *> & cutNtk, std::vector <boost::dynamic_bitset<ull>> & bdCut2Node101, std::vector <boost::dynamic_bitset<ull>> & bdCut2Node110, std::vector <boost::dynamic_bitset<ull>> & bdCut2Node011, std::vector <boost::dynamic_bitset<ull>> & bdCut2Node111);
};

bool IsPIOSame(Simulator & smlt0, Simulator & smlt1);
bool IsPIOSame(NetMan & net0, NetMan & net1);

bigInt GetDecOut(const boost::dynamic_bitset <ull> & poValues, ll nPo, bool isSign);   // get decimal output
