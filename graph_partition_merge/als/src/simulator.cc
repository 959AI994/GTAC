#include "simulator.h"


using namespace abc;
using namespace std;
using namespace boost;
using namespace random;


Simulator::Simulator(NetMan & net_man, unsigned _seed, ll n_frame): NetMan(net_man.GetNet(), false), seed(_seed), nFrame(n_frame) {
    #ifdef DEBUG
    auto type = NetMan::GetNetType();
    assert(type == NET_TYPE::AIG || type == NET_TYPE::GATE || type == NET_TYPE::SOP);
    #endif
    dat.resize(NetMan::GetIdMaxPlus1(), dynamic_bitset <ull> (nFrame, 0));
}


void Simulator::InpUnif() {
    uniform_int <> unif01(0, 1);
    random::mt19937 eng(seed);
    variate_generator < random::mt19937, uniform_int <> > rand01(eng, unif01);

    for (ll i = 0; i < NetMan::GetPiNum(); ++i) {
        auto piId = NetMan::GetPiId(i);
        dat[piId].reset();
        for (ll j = 0; j < nFrame; ++j) {
            if (rand01())
                dat[piId].set(j);
        }
    }
    
    for (ll i = 0; i < NetMan::GetIdMaxPlus1(); ++i) {
        if (NetMan::IsConst0(i))
            dat[i].reset();
        else if (NetMan::IsConst1(i))
            dat[i].set();
    }
}


void Simulator::InpUnifFast() {
    random::uniform_int_distribution <ull> unifUll;
    random::mt19937 eng(seed);
    variate_generator <random::mt19937, random::uniform_int_distribution <ull> > randUll(eng, unifUll);
    const ll unitLength = 64;
    assert((nFrame & (unitLength - 1)) == 0);
    ll nUnit = nFrame / unitLength;

    for (ll i = 0; i < NetMan::GetPiNum(); ++i) {
        auto piId = NetMan::GetPiId(i);
        dat[piId].resize(0);
        for (ll j = 0; j < nUnit; ++j) {
            ull numb = randUll();
            dat[piId].append(numb);
        }
    }

    for (ll i = 0; i < NetMan::GetIdMaxPlus1(); ++i) {
        if (NetMan::IsConst0(i))
            dat[i].reset();
        else if (NetMan::IsConst1(i))
            dat[i].set();
    }
}


void Simulator::InpEnum() {
    #ifdef DEBUG
    assert(GetPiNum() < 30);
    assert(1ll << GetPiNum() == nFrame);
    #endif
    for (ll i = 0; i < GetPiNum(); ++i) {
        bool phase = 1;
        auto piId = GetPiId(i);
        dat[piId].reset();
        for (ll j = 0; j < nFrame; ++j) {
            if (j % (1 << i) == 0)
                phase = !phase;
            if (phase)
                dat[piId].set(j);
        }
    }

    for (ll i = 0; i < NetMan::GetIdMaxPlus1(); ++i) {
        if (NetMan::IsConst0(i))
            dat[i].reset();
        else if (NetMan::IsConst1(i))
            dat[i].set();
    }
}


void Simulator::InpMix() {
    // only for 9x8 multiplier
    const ll inWidth = 9;
    const ll wWidth = 8;
    const double inMean = 0.0;
    const double inStandDer = 85.0;

    assert(inWidth >= 1 && inWidth <= 16);
    assert(wWidth >= 1 && wWidth <= 16);
    const double inMax = (1ll << (inWidth - 1)) - 1;
    const ll wMax = (1ll << (wWidth - 1)) - 1;
    assert(GetPiNum() == inWidth + wWidth);
    assert(inMean >= -inMax && inMean <= inMax);

    // "in" follows normal distribution
    boost::random::mt19937 engine0(seed);
    boost::random::normal_distribution <double> gauss(inMean, inStandDer);
    boost::variate_generator <boost::random::mt19937, boost::random::normal_distribution <double> > randIn(engine0, gauss);

    // "w" follows uniform distribution
    boost::random::mt19937 engine1(seed);
    boost::random::uniform_int_distribution <ll> unif(-wMax, wMax);
    boost::variate_generator <boost::random::mt19937, boost::random::uniform_int_distribution <ll> > randW(engine1, unif);

    // primary inputs
    abc::Abc_Obj_t * pObj = nullptr;
    ll k = 0;
    for (ll i = 0; i < nFrame; ++i) {
        ll in = static_cast <ll> (std::min(std::max(round(randIn()), -inMax), inMax));
        ll w = randW();
        auto in2Comp = bitset <16> ((in >= 0)? in: (1ll << inWidth) + in);
        auto w2Comp = bitset <16> ((w >= 0)? w: (1ll << wWidth) + w);
        for (k = 0; k < inWidth; ++k) {
            pObj = GetPi(k);
            dat[pObj->Id][i] = in2Comp[k];
        }
        for (k = inWidth; k < inWidth + wWidth; ++k) {
            pObj = GetPi(k);
            dat[pObj->Id][i] = w2Comp[k - inWidth];
        }
        // cout << in << "," << GetInput(i, 0, inWidth - 1, true) << endl;
        // cout << w << "," << GetInput(i, inWidth, inWidth + wWidth - 1, true) << endl;
        assert(in == GetInp(i, 0, inWidth - 1, true) && w == GetInp(i, inWidth, inWidth + wWidth - 1, true));
    }

    // constant nodes
    for (ll i = 0; i < NetMan::GetIdMaxPlus1(); ++i) {
        if (NetMan::IsConst0(i))
            dat[i].reset();
        else if (NetMan::IsConst1(i))
            dat[i].set();
    }
}


void Simulator::InpSelf(const string & fileName) {
    // primary inputs
    FILE * fp = fopen(fileName.c_str(), "r");
    assert(fp != nullptr);
    const ll maxPiNumb = 1000;
    assert(GetPiNum() <= maxPiNumb);
    char buf[maxPiNumb];
    ll cnt = 0;
    while (fgets(buf, sizeof(buf), fp) != nullptr) {
        assert(static_cast <ll>(strlen(buf)) == GetPiNum() + 1);
        for (ll i = 0; i < GetPiNum(); ++i) {
            auto pObj = GetPi(i);
            dat[pObj->Id].set(cnt, buf[i] == '1');
        }
        ++cnt;
        assert(cnt <= nFrame);
    }
    assert(cnt == nFrame);
    fclose(fp);

    // constant nodes
    for (ll i = 0; i < NetMan::GetIdMaxPlus1(); ++i) {
        if (NetMan::IsConst0(i))
            dat[i].reset();
        else if (NetMan::IsConst1(i))
            dat[i].set();
    }
}


void Simulator::Sim() {
    auto type = GetNetType();
    auto nodes = TopoSort();
    for (const auto & pObj: nodes) {
        if (type == NET_TYPE::AIG)
            UpdAigNode(pObj);
        else if (type == NET_TYPE::SOP)
            UpdSopNode(pObj);
        else if (type == NET_TYPE::GATE)
            UpdGateNode(pObj);
        else
            assert(0);
    }
    for (ll i = 0; i < GetPoNum(); ++i) {
        auto pPo = GetPo(i);
        auto drivId = GetFaninId(pPo, 0);
        #ifdef DEBUG
        assert(!Abc_ObjIsComplement(pPo));
        #endif
        dat[GetId(pPo)] = dat[drivId];
    }
}


void Simulator::UpdAigNode(Abc_Obj_t * pObj) {
    #ifdef DEBUG
    assert(Abc_ObjIsNode(pObj));
    #endif
    auto pNtk = NetMan::GetNet();
    auto pMan = static_cast <Hop_Man_t *> (pNtk->pManFunc);
    auto pRoot = static_cast <Hop_Obj_t *> (pObj->pData);
    auto pRootR = Hop_Regular(pRoot);

    // skip constant node
    if (Hop_ObjIsConst1(pRootR))
        return;

    // get topological order of subnetwork in aig
    Vec_Ptr_t * vHopNodes = Hop_ManDfsNode(pMan, pRootR);

    // init internal hop nodes
    ll maxHopId = -1;
    ll i = 0;
    Hop_Obj_t * pHopObj = nullptr;
    Vec_PtrForEachEntry(Hop_Obj_t *, vHopNodes, pHopObj, i)
        maxHopId = max(maxHopId, static_cast <ll> (pHopObj->Id));
    Vec_PtrForEachEntry( Hop_Obj_t *, pMan->vPis, pHopObj, i )
        maxHopId = max(maxHopId, static_cast <ll> (pHopObj->Id));
    vector < dynamic_bitset <ull> > interData(maxHopId + 1, dynamic_bitset <ull> (nFrame, 0));
    unordered_map <ll, dynamic_bitset <ull> *> hop2Data;
    Abc_Obj_t * pFanin = nullptr;
    Abc_ObjForEachFanin(pObj, pFanin, i)
        hop2Data[Hop_ManPi(pMan, i)->Id] = &dat[pFanin->Id];

    // special case for inverter or buffer
    if (pRootR->Type == AIG_PI) {
        pFanin = Abc_ObjFanin0(pObj);
        dat[pObj->Id] = dat[pFanin->Id];
    }

    // simulate
    Vec_PtrForEachEntry(Hop_Obj_t *, vHopNodes, pHopObj, i) {
        assert(Hop_ObjIsAnd(pHopObj));
        auto pHopFanin0 = Hop_ObjFanin0(pHopObj);
        auto pHopFanin1 = Hop_ObjFanin1(pHopObj);
        #ifdef DEBUG
        assert(!Hop_ObjIsConst1(pHopFanin0));
        assert(!Hop_ObjIsConst1(pHopFanin1));
        #endif
        dynamic_bitset <ull> & data0 = Hop_ObjIsPi(pHopFanin0) ? *hop2Data[pHopFanin0->Id] : interData[pHopFanin0->Id];
        dynamic_bitset <ull> & data1 = Hop_ObjIsPi(pHopFanin1) ? *hop2Data[pHopFanin1->Id] : interData[pHopFanin1->Id];
        dynamic_bitset <ull> & out = (pHopObj == pRootR) ? dat[pObj->Id] : interData[pHopObj->Id];
        bool isFanin0C = Hop_ObjFaninC0(pHopObj);
        bool isFanin1C = Hop_ObjFaninC1(pHopObj);
        if (!isFanin0C && !isFanin1C)
            out = data0 & data1;
        else if (!isFanin0C && isFanin1C)
            out = data0 & ~data1;
        else if (isFanin0C && !isFanin1C)
            out = ~data0 & data1;
        else if (isFanin0C && isFanin1C)
            out = ~(data0 | data1);
    }

    // complement
    if (Hop_IsComplement(pRoot))
        dat[pObj->Id].flip();

    // recycle memory
    Vec_PtrFree(vHopNodes); 
}


void Simulator::UpdSopNode(Abc_Obj_t * pObj) {
    #ifdef DEBUG
    assert(Abc_ObjIsNode(pObj));
    #endif
    // skip constant node
    if (Abc_NodeIsConst(pObj))
        return;
    // update sop
    char * pSop = static_cast <char *> (pObj->pData);
    UpdSop(pObj, pSop);
}


void Simulator::UpdGateNode(Abc_Obj_t * pObj) {
    #ifdef DEBUG
    assert(Abc_ObjIsNode(pObj));
    #endif
    // skip constant node
    if (Abc_NodeIsConst(pObj))
        return;
    // update sop
    char * pSop = static_cast <char *> ((static_cast <Mio_Gate_t *> (pObj->pData))->pSop);
    UpdSop(pObj, pSop);
}


void Simulator::UpdSop(Abc_Obj_t * pObj, char * pSop) {
    ll nVars = Abc_SopGetVarNum(pSop);
    dynamic_bitset <ull> product(nFrame, 0);
    for (char * pCube = pSop; *pCube; pCube += nVars + 3) {
        bool isFirst = true;
        for (ll i = 0; pCube[i] != ' '; i++) {
            Abc_Obj_t * pFanin = Abc_ObjFanin(pObj, i);
            switch (pCube[i]) {
                case '-':
                    continue;
                    break;
                case '0':
                    if (isFirst) {
                        isFirst = false;
                        product = ~dat[pFanin->Id];
                    }
                    else
                        product &= ~dat[pFanin->Id];
                    break;
                case '1':
                    if (isFirst) {
                        isFirst = false;
                        product = dat[pFanin->Id];
                    }
                    else
                        product &= dat[pFanin->Id];
                    break;
                default:
                    assert(0);
            }
        }
        if (isFirst) {
            isFirst = false;
            product.set();
        }
        #ifdef DEBUG
        assert(!isFirst);
        #endif
        if (pCube == pSop)
            dat[pObj->Id] = product;
        else
            dat[pObj->Id] |= product;
    }

    // complement
    if (Abc_SopIsComplement(pSop))
        dat[pObj->Id].flip();
}


bigInt Simulator::GetInp(ll iPatt, ll lsb, ll msb, bool isSign) const {
    #ifdef DEBUG
    assert(lsb >= 0 && msb < NetMan::GetPiNum());
    assert(iPatt < nFrame);
    assert(lsb <= msb && msb - lsb < 512);
    #endif
    bigInt ret(0);
    for (ll k = msb; k >= lsb; --k) {
        ret <<= 1;
        if (dat[NetMan::GetPiId(k)][iPatt])
            ++ret;
    }
    if (isSign && ret >= (static_cast <bigInt> (1) << (msb - lsb)))
        ret = -((static_cast <bigInt> (1) << (msb - lsb + 1)) - ret);
    return ret;
}


void Simulator::PrintInpStream(ll iPatt, bool isRev) const {
    #ifdef DEBUG
    assert(iPatt < nFrame);
    #endif
    if (isRev) {
        for (ll k = GetPiNum() - 1; k >= 0; --k)
            cout << dat[GetPiId(k)][iPatt];
    }
    else {
        for (ll k = 0; k < GetPiNum(); ++k)
            cout << dat[GetPiId(k)][iPatt];
    }
    cout << endl;
}


bigInt Simulator::GetOutp(ll iPatt) const {
    ll lsb = 0;
    ll msb = NetMan::GetPoNum() - 1;
    #ifdef DEBUG
    assert(iPatt < nFrame);
    assert(msb < 512);
    #endif
    bigInt ret(0);
    for (ll k = msb; k >= lsb; --k) {
        ret <<= 1;
        if (dat[NetMan::GetPoId(k)][iPatt])
            ++ret;
    }
    return ret;
}


bigInt Simulator::GetOutpPro(ll iPatt, bool isSign) const {
    ll lsb = 0;
    ll msb = NetMan::GetPoNum() - 1;
    ll shift = msb - lsb;
    assert(iPatt < nFrame);
    bigInt ret(0);
    for (ll k = msb; k >= lsb; --k) {
        ret <<= 1;
        if (dat[NetMan::GetPoId(k)][iPatt])
            ++ret;
    }
    if (isSign && ret >= (bigInt(1) << shift))
        ret = -((bigInt(1) << (shift + 1)) - ret);
    return ret;
}

vector <bigInt> Simulator::GetOutpPro_complex(ll iPatt, bool isSign , ll nOutput) const {
    ll lsb = 0;
    ll msb = NetMan::GetPoNum()/nOutput - 1;
    ll shift = msb - lsb;
    vector <bigInt> ret(nOutput,0);
    // cout << "ret.size = " << ret.size() << endl;
    // cout << "ret(0) = " << ret[0] << ", ret(1) = " << ret[1] << endl;
    assert(iPatt < nFrame);
    for (auto divnum = 0; divnum < nOutput; ++divnum){
        for (ll k = msb + divnum * (msb + 1); k >= lsb + divnum * (msb + 1); --k) {
            ret[divnum] <<= 1;
            if (dat[NetMan::GetPoId(k)][iPatt])
                ++ret[divnum];
        }
        if (isSign && ret[divnum] >= (bigInt(1) << shift))
            ret[divnum] = -((bigInt(1) << (shift + 1)) - ret[divnum]);
    }
    return ret;
}


bigInt Simulator::GetTempOutpPro(ll iPatt, bool isSign) const {
    ll lsb = 0;
    ll msb = NetMan::GetPoNum() - 1;
    ll shift = msb - lsb;
    assert(iPatt < nFrame);
    bigInt ret(0);
    for (ll k = msb; k >= lsb; --k) {
        ret <<= 1;
        if (tempDat[NetMan::GetPoId(k)][iPatt])
            ++ret;
    }
    if (isSign && ret >= (bigInt(1) << shift))
        ret = -((bigInt(1) << (shift + 1)) - ret);
    return ret;
}


ll Simulator::GetOutpFast(ll iPatt, bool isSign) const {
    ll lsb = 0;
    ll msb = NetMan::GetPoNum() - 1;
    #ifdef DEBUG
    assert(iPatt < nFrame);
    assert(msb < 60);
    #endif
    ll ret = 0;
    for (ll k = msb; k >= lsb; --k) {
        ret <<= 1;
        if (dat[NetMan::GetPoId(k)][iPatt])
            ++ret;
    }
    if (isSign && ret >= (1ll << (msb - lsb)))
        ret = -((1ll << (msb - lsb + 1)) - ret);
    return ret;
}


ll Simulator::GetTempOutpFast(ll iPatt, bool isSign) const {
    ll lsb = 0;
    ll msb = NetMan::GetPoNum() - 1;
    #ifdef DEBUG
    assert(iPatt < nFrame);
    assert(msb < 60);
    #endif
    ll ret = 0;
    for (ll k = msb; k >= lsb; --k) {
        ret <<= 1;
        if (tempDat[NetMan::GetPoId(k)][iPatt])
            ++ret;
    }
    if (isSign && ret >= (1ll << (msb - lsb)))
        ret = -((1ll << (msb - lsb + 1)) - ret);
    return ret;
}


void Simulator::PrintOutpStream(ll iPatt) const {
    #ifdef DEBUG
    assert(iPatt < nFrame);
    #endif
    for (ll k = GetPoNum() - 1; k >= 0; --k)
        cout << dat[GetPoId(k)][iPatt];
    cout << endl;
}


double Simulator::GetSignalProb(ll objId) const {
    #ifdef DEBUG
    assert(objId < NetMan::GetIdMaxPlus1());
    #endif
    return dat[objId].count() / static_cast <double> (nFrame);
}


void Simulator::PrintSignalProb() const {
    for (ll i = 0; i < NetMan::GetPoNum(); ++i) {
        cout << NetMan::GetName(NetMan::GetPo(i)) << " " << GetSignalProb(NetMan::GetPoId(i)) << endl;
    }
}


bool Simulator::IsPIOSame(const Simulator & oth_smlt) const {
    if (this->GetPiNum() != oth_smlt.GetPiNum())
        return false;
    for (ll i = 0; i < this->GetPiNum(); ++i) {
        // if (strcmp(Abc_ObjName(Abc_NtkPo(pNtk1, i)), Abc_ObjName(Abc_NtkPo(pNtk2, i))) != 0)
        if (this->GetPiName(i) != oth_smlt.GetPiName(i))
            return false;
    }
    if (this->GetPoNum() != oth_smlt.GetPoNum())
        return false;
    for (ll i = 0; i < this->GetPoNum(); ++i) {
        // if (strcmp(Abc_ObjName(Abc_NtkPo(pNtk1, i)), Abc_ObjName(Abc_NtkPo(pNtk2, i))) != 0)
        if (this->GetPoName(i) != oth_smlt.GetPoName(i))
            return false;
    }
    return true;
}


double Simulator::GetErrRate(const Simulator & oth_smlt, bool isSign, ll nOutput, vector <ll> RealCom, ll cutId, bool isCheck) const {
    // if (isCheck) {
    //     assert(IsPIOSame(oth_smlt));
    // }
    // bigInt sed = 0;
    // for (ll i = 0; i < nFrame; ++i) {
    //     // // Single Form
    //     // ll accOut = GetOutpFast(i, isSign);
    //     // ll appOut = oth_smlt.GetOutpFast(i, isSign);
    //     // sed += abs(accOut - appOut);

    //     // Multiple Form
    //     auto accOut = GetOutpPro_complex(i, isSign, nOutput);
    //     auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
    //     for (ll j = 0; j < nOutput; ++j)
    //         sed += appOut[j] - accOut[j];
    // }
    // return double(bigFlt(sed) / bigFlt(nFrame));


    if (isCheck) {
        assert(IsPIOSame(oth_smlt));
    }
    // bigInt sed = 0;
    // int flag = 0;
    // for (ll i = 0; i < nFrame; ++i) {
    //     // Multiple Form
    //     auto accOut = GetOutpPro_complex(i, isSign, nOutput);
    //     auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
    //     bigInt tmpE = 0;

    //     if (cutId == -1){
    //         for (ll j = 0; j < nOutput; ++j)
    //             tmpE += appOut[j] - accOut[j];
    //     }
    //     else {
    //         tmpE += appOut[cutId] - accOut[cutId];
    //     }
    //     if (tmpE == 0)
    //         continue;
    //     if (flag != 0 && tmpE * flag < 0) 
    //         return 0;
    //     else{  //flag == 0 || tmpE * flag >= 0
    //         if (tmpE > 0)
    //             flag = 1;
    //         else 
    //             flag = -1;
    //     }
    // }
    // return double(flag);

    ll num = 0;
    for (ll i = 0; i < nFrame; ++i) {
        auto accOut = GetOutpPro_complex(i, isSign, nOutput);
        auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
        bool flag = false;
        for (ll j = 0; j < nOutput; ++j) {
            if (accOut[j] != appOut[j]) {
                flag = true;
                break;
            }
        }
        if (flag)
            num++;
    }
    // cout << "simulation: num = " << num << ", nFrame = " << nFrame << endl;
    return double(bigFlt(num) / bigFlt(nFrame));
}


double Simulator::GetMeanErrDist(const Simulator & oth_smlt, bool isSign, ll nOutput, vector <ll> RealCom, ll cutId, bool isCheck) const {
    if (isCheck) {
        assert(IsPIOSame(oth_smlt));
    }
    bigInt sed = 0;
    if (cutId == -1) {
        for (ll i = 0; i < nFrame; ++i) {
            // // Single Form
            // ll accOut = GetOutpFast(i, isSign);
            // ll appOut = oth_smlt.GetOutpFast(i, isSign);
            // sed += abs(accOut - appOut);

            // Multiple Form
            auto accOut = GetOutpPro_complex(i, isSign, nOutput);
            auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
            for (ll j = 0; j < nOutput; ++j)
                sed += abs(appOut[j] - accOut[j]);
        }
    }
    else {
        for (ll i = 0; i < nFrame; ++i) {
            auto accOut = GetOutpPro_complex(i, isSign, nOutput);
            auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
            sed += abs(appOut[cutId] - accOut[cutId]);
        }
    }
    return double(bigFlt(sed) / bigFlt(nFrame));
}


double Simulator::GetMeanErr(const Simulator & oth_smlt, bool isSign, ll nOutput, vector <ll> RealCom, ll cutId, bool isCheck) const {
    if (isCheck) {
        assert(IsPIOSame(oth_smlt));
    }
    bigInt sed = 0;
    if (cutId == -1){
        for (ll i = 0; i < nFrame; ++i) {
            auto accOut = GetOutpPro_complex(i, isSign, nOutput);
            auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
            for (ll j = 0; j < nOutput; ++j)
                sed += accOut[j] - appOut[j];
        }
    }
    else {
        for (ll i = 0; i < nFrame; ++i) {
            auto accOut = GetOutpPro_complex(i, isSign, nOutput);
            auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
            sed += accOut[cutId] - appOut[cutId];
        }
    }
    return double(bigFlt(sed) / bigFlt(nFrame));
}


double Simulator::GetMeanSquareErr(const Simulator & oth_smlt, bool isSign, ll nOutput, vector <ll> RealCom, ll cutId, bool isCheck) const {
    if (isCheck) {
        assert(IsPIOSame(oth_smlt));
    }
    bigInt sse = 0;
    if (cutId == -1){
        for (ll i = 0; i < nFrame; ++i) {
            auto accOut = GetOutpPro_complex(i, isSign, nOutput);
            auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
            for (ll j = 0; j < nOutput; ++j){
                appOut[j] += RealCom[j];
                sse += (accOut[j] - appOut[j]) * (accOut[j] - appOut[j]);
            }
        }
    }
    else {
        for (ll i = 0; i < nFrame; ++i) {
            auto accOut = GetOutpPro_complex(i, isSign, nOutput);
            auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
            appOut[cutId] += RealCom[cutId];
            sse += (accOut[cutId] - appOut[cutId]) * (accOut[cutId] - appOut[cutId]);
        }
    }
    return double(bigFlt(sse) / bigFlt(nFrame));
}

double Simulator::GetMeanSquareErr_forDebug(const Simulator & oth_smlt, bool isSign, ll nOutput, vector <ll> RealCom, ll cutId, bool isCheck) const {
    if (isCheck) {
        assert(IsPIOSame(oth_smlt));
    }
    bigInt sse = 0;
    if (cutId == -1){
        for (ll i = 0; i < nFrame; ++i) {
            auto accOut = GetOutpPro_complex(i, isSign, nOutput);
            auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);

            // debug
            cout << appOut[0] << ", ";
            if ((i + 1) % 10 == 0)
                cout << endl;

            for (ll j = 0; j < nOutput; ++j){
                appOut[j] += RealCom[j];
                sse += (accOut[j] - appOut[j]) * (accOut[j] - appOut[j]);
            }
        }
    }
    else {
        for (ll i = 0; i < nFrame; ++i) {
            auto accOut = GetOutpPro_complex(i, isSign, nOutput);
            auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
            appOut[cutId] += RealCom[cutId];
            sse += (accOut[cutId] - appOut[cutId]) * (accOut[cutId] - appOut[cutId]);
        }
    }
    return double(bigFlt(sse) / bigFlt(nFrame));
}


double Simulator::GetSigNoiseRat(const Simulator & oth_smlt, bool isSign, ll nOutput, vector <ll> RealCom, ll cutId, bool isCheck) const {
    if (isCheck) {
        assert(IsPIOSame(oth_smlt));
    }
    bigInt sse = 0;
    bigInt sumAcc2 = 0;
    if (cutId == -1){
        for (ll i = 0; i < nFrame; ++i) {
            auto accOut = GetOutpPro_complex(i, isSign, nOutput);
            auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
            for (ll j = 0; j < nOutput; ++j){
                appOut[j] += RealCom[j];
                sse += (accOut[j] - appOut[j]) * (accOut[j] - appOut[j]);
                sumAcc2 += accOut[j] * accOut[j];
            }
        }
    }
    else {
        for (ll i = 0; i < nFrame; ++i) {
            auto accOut = GetOutpPro_complex(i, isSign, nOutput);
            auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
            appOut[cutId] += RealCom[cutId];
            sse += (accOut[cutId] - appOut[cutId]) * (accOut[cutId] - appOut[cutId]);
            sumAcc2 += accOut[cutId] * accOut[cutId];
        }
    }
    if (sse != 0) {
        auto rat = bigFlt(sumAcc2) / bigFlt(sse);
        return double(bigFlt(10) * log10(rat));
    }
    return numeric_limits <double>::max();
}

double Simulator::GetMaxErrDist(const Simulator & oth_smlt, bool isSign, ll nOutput, vector <ll> RealCom, ll cutId, bool isCheck) const {
    if (isCheck) {
        assert(IsPIOSame(oth_smlt));
    }
    bigInt sed = 0;
    for (ll i = 0; i < nFrame; ++i) {
        auto accOut = GetOutpPro_complex(i, isSign, nOutput);
        auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
        if (cutId == -1){
            bigInt tmpED = 0;
            for (ll j = 0; j < nOutput; ++j){
                appOut[j] += RealCom[j];
                tmpED += abs(appOut[j] - accOut[j]);
            }
            if (tmpED > sed)
                sed = tmpED;
        }
        else {
            bigInt tmpED = 0;
            appOut[cutId] += RealCom[cutId];
            tmpED += abs(appOut[cutId] - accOut[cutId]);
            if (tmpED > sed)
                sed = tmpED;
        }
    }
    return double(sed);
}

double Simulator::GetMeanRelErrDist(const Simulator & oth_smlt, bool isSign, ll nOutput, vector <ll> RealCom, ll cutId, bool isCheck) const {
    if (isCheck) {
        assert(IsPIOSame(oth_smlt));
    }
    bigFlt sum(0);
    if (cutId == -1) {
        for (ll i = 0; i < nFrame; ++i) {
            // Multiple Form
            auto accOut = GetOutpPro_complex(i, isSign, nOutput);
            auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
            for (ll j = 0; j < nOutput; ++j) {
                if (accOut[j] != 0)
                    sum += abs(1 - static_cast <bigFlt>(appOut[j])/static_cast <bigFlt>(accOut[j]));
                else
                    // sum += abs(1 - static_cast <bigFlt>(appOut[j]));
                    sum += abs(static_cast <bigFlt>(appOut[j]));
            }
        }
    }
    else {
        for (ll i = 0; i < nFrame; ++i) {
            auto accOut = GetOutpPro_complex(i, isSign, nOutput);
            auto appOut = oth_smlt.GetOutpPro_complex(i, isSign, nOutput);
            if (accOut[cutId] != 0)
                sum += abs(1 - static_cast <bigFlt>(appOut[cutId])/static_cast <bigFlt>(accOut[cutId]));
            else
                sum += abs(1 - static_cast <bigFlt>(appOut[cutId]));
        }
    }
    return double(sum / bigFlt(nFrame));
}


double Simulator::GetMeanHamDist(Simulator & oth_smlt, bool isSign, ll nOutput, vector <ll> RealCom, ll cutId, bool isCheck) {
    if (isCheck) {
        assert(IsPIOSame(oth_smlt));
    }
    ll nPo = GetPoNum();
    bigInt num = 0;
    for (ll i = 0; i < nFrame; ++i) {
        for (ll o = 0; o < nPo; ++o) {
            ll poId = GetPoId(o);
            if (GetDat(poId, i) != oth_smlt.GetDat(poId, i))
                ++num;
        }
    }
    return double(bigFlt(num) / bigFlt(nFrame));
}


// double Simulator::GetSelfDefErr(const Simulator & oth_smlt, bool isSign, const string & selfDefMetr, bool isCheck) const {
//     if (isCheck) {
//         assert(IsPIOSame(oth_smlt));
//         assert(GetPoNum() < 64);
//     }
//     std::string program;
//     std::vector <ll> outAcc(nFrame);
//     std::vector <ll> outApp(nFrame);
//     for (ll i = 0; i < nFrame; ++i) {
//         outAcc[i] = GetOutpFast(i, isSign);
//         outApp[i] = oth_smlt.GetOutpFast(i, isSign); 
//     }
//     AppendData(program, nFrame, outAcc, outApp);
//     program.append(selfDefMetr);
//     // cout << program << endl;
//     double err = GetErrFromExprtk(program);
//     // cout << err << endl;
//     return err;
// }


void Simulator::CalcLocBoolDiff(Abc_Obj_t * pObj, list <Abc_Obj_t *> & disjCut, vector <Abc_Obj_t *> & cutNtk, vector < dynamic_bitset <ull> > & bdCut2Node) {
    #ifdef DEBUG
    assert(pObj->pNtk == GetNet());
    #endif
    if (tempDat.size() != dat.size())
        tempDat.resize(dat.size(), dynamic_bitset <ull>(nFrame, 0));
    // flip the node
    tempDat[pObj->Id] = ~dat[pObj->Id];
    // simulate
    Abc_NtkIncrementTravId(GetNet());
    Abc_NodeSetTravIdCurrent(pObj);
    for (auto & pInner: cutNtk)
        Abc_NodeSetTravIdCurrent(pInner);
    auto type = NetMan::GetNetType();
    if (type == NET_TYPE::AIG)
        assert(0);
        // UpdAigNodeForBoolAndPartDiff(pObj);
    else if (type == NET_TYPE::SOP) {
        for (auto & pInner: cutNtk)
            UpdSopNodeForBoolAndPartDiff(pInner);
    }
    else if (type == NET_TYPE::GATE) {
        for (auto & pInner: cutNtk)
            UpdGateNodeForBoolAndPartDiff(pInner);
    }
    else
        assert(0);
    // get boolean difference from the node to its disjoint cuts
    bdCut2Node.resize(disjCut.size(), dynamic_bitset <ull>(nFrame, 0));
    ll i = 0;
    for (auto & pCut: disjCut) {
        bdCut2Node[i] = dat[pCut->Id] ^ tempDat[pCut->Id];
        ++i;
    }
}


void Simulator::CalcLocPartDiff(Abc_Obj_t * pObj, list <Abc_Obj_t *> & disjCut, vector <Abc_Obj_t *> & cutNtk, vector < vector <int8_t> > & pdCut2Node) {
    #ifdef DEBUG
    assert(pObj->pNtk == GetNet());
    #endif
    if (tempDat.size() != dat.size())
        tempDat.resize(dat.size(), dynamic_bitset <ull>(nFrame, 0));
    // flip the node
    tempDat[pObj->Id] = ~dat[pObj->Id];
    // simulate
    Abc_NtkIncrementTravId(GetNet());
    Abc_NodeSetTravIdCurrent(pObj);
    for (auto & pInner: cutNtk)
        Abc_NodeSetTravIdCurrent(pInner);
    auto type = NetMan::GetNetType();
    if (type == NET_TYPE::AIG)
        assert(0);
    else if (type == NET_TYPE::SOP) {
        for (auto & pInner: cutNtk)
            UpdSopNodeForBoolAndPartDiff(pInner);
    }
    else if (type == NET_TYPE::GATE) {
        for (auto & pInner: cutNtk)
            UpdGateNodeForBoolAndPartDiff(pInner);
    }
    else
        assert(0);
    // get boolean difference from the node to its disjoint cuts
    pdCut2Node.resize(disjCut.size(), vector <int8_t> (nFrame, 0));
    ll i = 0;
    // parallel acceleration
    for (auto & pCut: disjCut) {
        // if (useMP) {
        //     omp_set_num_threads(numOfThread);
        //     #pragma omp parallel for schedule(dynamic)
        //     for (ll j = 0; j < nFrame; ++j) {
        //         pdCut2Node[i][j] = ((int8_t)1 - ((int8_t)dat[pObj->Id][j] << int8_t(1))) * ((int8_t)tempDat[pCut->Id][j] - (int8_t)dat[pCut->Id][j]);
        //     }
        // }
        // else 
        {
            for (ll j = 0; j < nFrame; ++j) {
                pdCut2Node[i][j] = ((int8_t)1 - ((int8_t)dat[pObj->Id][j] << int8_t(1))) * ((int8_t)tempDat[pCut->Id][j] - (int8_t)dat[pCut->Id][j]);
            }
        }
        ++i;
    }
}


void Simulator::UpdSopNodeForBoolAndPartDiff(Abc_Obj_t * pObj) {
    #ifdef DEBUG
    assert(!Abc_ObjIsPi(pObj));
    assert(!Abc_NodeIsConst(pObj));
    #endif
    if (Abc_ObjIsPo(pObj)) {
        #ifdef DEBUG
        assert(!Abc_ObjIsComplement(pObj));
        #endif
        Abc_Obj_t * pDriver = Abc_ObjFanin0(pObj);
        if (Abc_NodeIsTravIdCurrent(pDriver))
            tempDat[pObj->Id] = tempDat[pDriver->Id];
        else
            tempDat[pObj->Id] = dat[pDriver->Id];
        return;
    }
    // update sop
    char * pSop = static_cast <char *> (pObj->pData);
    UpdSopForBoolAndPartDiff(pObj, pSop);
}


void Simulator::UpdGateNodeForBoolAndPartDiff(Abc_Obj_t * pObj) {
    #ifdef DEBUG
    assert(!Abc_ObjIsPi(pObj));
    assert(!Abc_NodeIsConst(pObj));
    #endif
    if (Abc_ObjIsPo(pObj)) {
        Abc_Obj_t * pDriver = Abc_ObjFanin0(pObj);
        if (Abc_NodeIsTravIdCurrent(pDriver))
            tempDat[pObj->Id] = tempDat[pDriver->Id];
        else
            tempDat[pObj->Id] = dat[pDriver->Id];
        return;
    }
    // update sop
    char * pSop = static_cast <char *> ((static_cast <Mio_Gate_t *> (pObj->pData))->pSop);
    UpdSopForBoolAndPartDiff(pObj, pSop);
}


void Simulator::UpdSopForBoolAndPartDiff(Abc_Obj_t * pObj, char * pSop) {
    ll nVars = Abc_SopGetVarNum(pSop);
    dynamic_bitset <ull> product(nFrame, 0);
    for (char * pCube = pSop; *pCube; pCube += nVars + 3) {
        bool isFirst = true;
        for (ll i = 0; pCube[i] != ' '; i++) {
            Abc_Obj_t * pFanin = Abc_ObjFanin(pObj, i);
            dynamic_bitset <ull> & datFi = Abc_NodeIsTravIdCurrent(pFanin)? tempDat[pFanin->Id]: dat[pFanin->Id];
            switch (pCube[i]) {
                case '-':
                    continue;
                    break;
                case '0':
                    if (isFirst) {
                        isFirst = false;
                        product = ~datFi;
                    }
                    else
                        product &= ~datFi;
                    break;
                case '1':
                    if (isFirst) {
                        isFirst = false;
                        product = datFi;
                    }
                    else
                        product &= datFi;
                    break;
                default:
                    assert(0);
            }
        }
        if (isFirst) {
            isFirst = false;
            product.set();
        }
        #ifdef DEBUG
        assert(!isFirst);
        #endif
        if (pCube == pSop) {
            tempDat[pObj->Id] = product;
        }
        else
            tempDat[pObj->Id] |= product;
    }

    // complement
    if (Abc_SopIsComplement(pSop))
        tempDat[pObj->Id].flip();
}


bool IsPIOSame(Simulator & smlt0, Simulator & smlt1) {
    if (smlt0.GetPiNum() != smlt1.GetPiNum())
        return false;
    for (ll i = 0; i < smlt0.GetPiNum(); ++i) {
        // if (strcmp(Abc_ObjName(Abc_NtkPo(pNtk1, i)), Abc_ObjName(Abc_NtkPo(pNtk2, i))) != 0)
        if (smlt0.GetPiName(i) != smlt1.GetPiName(i))
            return false;
    }
    if (smlt0.GetPoNum() != smlt1.GetPoNum())
        return false;
    for (ll i = 0; i < smlt0.GetPoNum(); ++i) {
        // if (strcmp(Abc_ObjName(Abc_NtkPo(pNtk1, i)), Abc_ObjName(Abc_NtkPo(pNtk2, i))) != 0)
        if (smlt0.GetPoName(i) != smlt1.GetPoName(i))
            return false;
    }
    return true;
}


bool IsPIOSame(NetMan & net0, NetMan & net1) {
    if (net0.GetPiNum() != net1.GetPiNum())
        return false;
    for (ll i = 0; i < net0.GetPiNum(); ++i) {
        // if (strcmp(Abc_ObjName(Abc_NtkPo(pNtk1, i)), Abc_ObjName(Abc_NtkPo(pNtk2, i))) != 0)
        if (net0.GetPiName(i) != net1.GetPiName(i))
            return false;
    }
    if (net0.GetPoNum() != net1.GetPoNum())
        return false;
    for (ll i = 0; i < net0.GetPoNum(); ++i) {
        // if (strcmp(Abc_ObjName(Abc_NtkPo(pNtk1, i)), Abc_ObjName(Abc_NtkPo(pNtk2, i))) != 0)
        if (net0.GetPoName(i) != net1.GetPoName(i))
            return false;
    }
    return true;
}

extern "C" {
    Vec_Ptr_t * Abc_MfsWinMarkTfi(Abc_Obj_t * pNode);
}
int Simulator::CalcLocBd2(ll objId1, ll objId2, list <Abc_Obj_t *> & disjCut, vector <Abc_Obj_t *> & cutNtk, vector <boost::dynamic_bitset<ull>> & bdCut2Node11, vector <boost::dynamic_bitset<ull>> & bdCut2Node10) {
    Abc_Obj_t * pObj1 = GetObj(objId1);
    Abc_Obj_t * pObj2 = GetObj(objId2);

    // check the relationship between the 2 nodes
    assert(objId1 < objId2);
    int fRelation = 0;     // 0: no TFI/TFO relationship, 1: pObj1 is a TFI of pObj2
    Abc_NtkIncrementTravId(GetNet());
    Abc_MfsWinMarkTfi(pObj2);
    if (Abc_NodeIsTravIdCurrent(pObj1))
        fRelation = 1;
    
    // if (tempDat.size() != dat.size())
    //     tempDat.resize(dat.size(), dynamic_bitset <ull>(nFrame, 0));
    tempDat.resize(dat.size(), dynamic_bitset <ull>(nFrame, 0));
    if (fRelation == 0) {
        // flip 2 nodes
        tempDat[objId1] = ~dat[objId1];
        tempDat[objId2] = ~dat[objId2];

        // simulate
        Abc_NtkIncrementTravId(GetNet());
        Abc_NodeSetTravIdCurrent(pObj1);
        Abc_NodeSetTravIdCurrent(pObj2);
        for (auto & pInner: cutNtk)
            Abc_NodeSetTravIdCurrent(pInner);
        auto type = NetMan::GetNetType();
        if (type == NET_TYPE::AIG)
            assert(0);
            // UpdAigNodeForBoolAndPartDiff(pObj);
        else if (type == NET_TYPE::SOP) {
            for (auto & pInner: cutNtk)
                UpdSopNodeForBoolAndPartDiff(pInner);
        }
        else if (type == NET_TYPE::GATE) {
            for (auto & pInner: cutNtk)
                UpdGateNodeForBoolAndPartDiff(pInner);
        }
        else
            assert(0);
        
        // get boolean difference from the node to its disjoint cuts
        bdCut2Node11.resize(disjCut.size(), dynamic_bitset <ull>(nFrame, 0));
        // bdCut2Node10.resize(disjCut.size(), dynamic_bitset <ull>(nFrame, 0));    // bdCut2Node10 is empty
        ll i = 0;
        for (auto & pCut: disjCut) {
            bdCut2Node11[i] = dat[pCut->Id] ^ tempDat[pCut->Id];
            ++i;
        }
    }
    else {
        assert(fRelation == 1);
        // 1. flip 2 nodes
        tempDat[objId1] = ~dat[objId1];
        tempDat[objId2] = ~dat[objId2];

        // simulate
        Abc_NtkIncrementTravId(GetNet());
        Abc_NodeSetTravIdCurrent(pObj1);
        Abc_NodeSetTravIdCurrent(pObj2);
        for (auto & pInner: cutNtk)
            Abc_NodeSetTravIdCurrent(pInner);
        auto type = NetMan::GetNetType();
        if (type == NET_TYPE::AIG)
            assert(0);
            // UpdAigNodeForBoolAndPartDiff(pObj);
        else if (type == NET_TYPE::SOP) {
            for (auto & pInner: cutNtk) {
                if (Abc_ObjId(pInner) == objId2)
                    continue;
                UpdSopNodeForBoolAndPartDiff(pInner);
            }
        }
        else if (type == NET_TYPE::GATE) {
            for (auto & pInner: cutNtk) {
                if (Abc_ObjId(pInner) == objId2)
                    continue;
                UpdGateNodeForBoolAndPartDiff(pInner);
            }
        }
        else
            assert(0);
        
        // get boolean difference from the node to its disjoint cuts
        bdCut2Node11.resize(disjCut.size(), dynamic_bitset <ull>(nFrame, 0));
        ll i = 0;
        for (auto & pCut: disjCut) {
            bdCut2Node11[i] = dat[pCut->Id] ^ tempDat[pCut->Id];
            ++i;
        }

        // 2. flip only one node
        tempDat[objId1] = ~dat[objId1];
        tempDat[objId2] = dat[objId2];

        // simulate
        Abc_NtkIncrementTravId(GetNet());
        Abc_NodeSetTravIdCurrent(pObj1);
        Abc_NodeSetTravIdCurrent(pObj2);
        for (auto & pInner: cutNtk)
            Abc_NodeSetTravIdCurrent(pInner);
        type = NetMan::GetNetType();
        if (type == NET_TYPE::AIG)
            assert(0);
            // UpdAigNodeForBoolAndPartDiff(pObj);
        else if (type == NET_TYPE::SOP) {
            for (auto & pInner: cutNtk) {
                if (Abc_ObjId(pInner) == objId2)
                    continue;
                UpdSopNodeForBoolAndPartDiff(pInner);
            }
        }
        else if (type == NET_TYPE::GATE) {
            for (auto & pInner: cutNtk) {
                if (Abc_ObjId(pInner) == objId2)
                    continue;
                UpdGateNodeForBoolAndPartDiff(pInner);
            }
        }
        else
            assert(0);

        // get boolean difference from the node to its disjoint cuts
        bdCut2Node10.resize(disjCut.size(), dynamic_bitset <ull>(nFrame, 0));
        i = 0;
        for (auto & pCut: disjCut) {
            bdCut2Node10[i] = dat[pCut->Id] ^ tempDat[pCut->Id];
            ++i;
        }
    }
    return fRelation;
}


void Simulator::CalcLocBd3(ll objId1, ll objId2, ll objId3, std::list <Abc_Obj_t *> & disjCut, std::vector <Abc_Obj_t *> & cutNtk, std::vector <boost::dynamic_bitset<ull>> & bdCut2Node101, std::vector <boost::dynamic_bitset<ull>> & bdCut2Node110, std::vector <boost::dynamic_bitset<ull>> & bdCut2Node011, std::vector <boost::dynamic_bitset<ull>> & bdCut2Node111) {
    Abc_Obj_t * pObj1 = GetObj(objId1);
    Abc_Obj_t * pObj2 = GetObj(objId2);
    Abc_Obj_t * pObj3 = GetObj(objId3);
    
    tempDat.resize(dat.size(), dynamic_bitset <ull>(nFrame, 0));
    
    // 1. flip node 1 and 3 (101)
    tempDat[objId1] = ~dat[objId1];
    tempDat[objId2] = dat[objId2];
    tempDat[objId3] = ~dat[objId3];

    // simulate
    Abc_NtkIncrementTravId(GetNet());
    Abc_NodeSetTravIdCurrent(pObj1);
    Abc_NodeSetTravIdCurrent(pObj2);
    Abc_NodeSetTravIdCurrent(pObj3);
    for (auto & pInner: cutNtk)
        Abc_NodeSetTravIdCurrent(pInner);
    auto type = NetMan::GetNetType();
    if (type == NET_TYPE::AIG)
        assert(0);
        // UpdAigNodeForBoolAndPartDiff(pObj);
    else if (type == NET_TYPE::SOP) {
        for (auto & pInner: cutNtk)
            UpdSopNodeForBoolAndPartDiff(pInner);
    }
    else if (type == NET_TYPE::GATE) {
        for (auto & pInner: cutNtk)
            UpdGateNodeForBoolAndPartDiff(pInner);
    }
    else
        assert(0);
    
    // get boolean difference from the node to its disjoint cuts
    bdCut2Node101.resize(disjCut.size(), dynamic_bitset <ull>(nFrame, 0));
    ll i = 0;
    for (auto & pCut: disjCut) {
        bdCut2Node101[i] = dat[pCut->Id] ^ tempDat[pCut->Id];
        ++i;
    }

    // --------------------------------------
    // 2. flip node 1 and 2 (110)
    tempDat[objId1] = ~dat[objId1];
    tempDat[objId2] = ~dat[objId2];
    tempDat[objId3] = dat[objId3];

    // simulate
    Abc_NtkIncrementTravId(GetNet());
    Abc_NodeSetTravIdCurrent(pObj1);
    Abc_NodeSetTravIdCurrent(pObj2);
    Abc_NodeSetTravIdCurrent(pObj3);
    for (auto & pInner: cutNtk)
        Abc_NodeSetTravIdCurrent(pInner);
    type = NetMan::GetNetType();
    if (type == NET_TYPE::AIG)
        assert(0);
        // UpdAigNodeForBoolAndPartDiff(pObj);
    else if (type == NET_TYPE::SOP) {
        for (auto & pInner: cutNtk)
            UpdSopNodeForBoolAndPartDiff(pInner);
    }
    else if (type == NET_TYPE::GATE) {
        for (auto & pInner: cutNtk)
            UpdGateNodeForBoolAndPartDiff(pInner);
    }
    else
        assert(0);
    
    // get boolean difference from the node to its disjoint cuts
    bdCut2Node110.resize(disjCut.size(), dynamic_bitset <ull>(nFrame, 0));
    i = 0;
    for (auto & pCut: disjCut) {
        bdCut2Node110[i] = dat[pCut->Id] ^ tempDat[pCut->Id];
        ++i;
    }

    // --------------------------------------
    // 3. flip node 2 and 3 (011)
    tempDat[objId1] = dat[objId1];
    tempDat[objId2] = ~dat[objId2];
    tempDat[objId3] = ~dat[objId3];

    // simulate
    Abc_NtkIncrementTravId(GetNet());
    Abc_NodeSetTravIdCurrent(pObj1);
    Abc_NodeSetTravIdCurrent(pObj2);
    Abc_NodeSetTravIdCurrent(pObj3);
    for (auto & pInner: cutNtk)
        Abc_NodeSetTravIdCurrent(pInner);
    type = NetMan::GetNetType();
    if (type == NET_TYPE::AIG)
        assert(0);
        // UpdAigNodeForBoolAndPartDiff(pObj);
    else if (type == NET_TYPE::SOP) {
        for (auto & pInner: cutNtk)
            UpdSopNodeForBoolAndPartDiff(pInner);
    }
    else if (type == NET_TYPE::GATE) {
        for (auto & pInner: cutNtk)
            UpdGateNodeForBoolAndPartDiff(pInner);
    }
    else
        assert(0);
    
    // get boolean difference from the node to its disjoint cuts
    bdCut2Node011.resize(disjCut.size(), dynamic_bitset <ull>(nFrame, 0));
    i = 0;
    for (auto & pCut: disjCut) {
        bdCut2Node011[i] = dat[pCut->Id] ^ tempDat[pCut->Id];
        ++i;
    }

    // --------------------------------------
    // 4. flip node 1, 2 and 3 (111)
    tempDat[objId1] = ~dat[objId1];
    tempDat[objId2] = ~dat[objId2];
    tempDat[objId3] = ~dat[objId3];

    // simulate
    Abc_NtkIncrementTravId(GetNet());
    Abc_NodeSetTravIdCurrent(pObj1);
    Abc_NodeSetTravIdCurrent(pObj2);
    Abc_NodeSetTravIdCurrent(pObj3);
    for (auto & pInner: cutNtk)
        Abc_NodeSetTravIdCurrent(pInner);
    type = NetMan::GetNetType();
    if (type == NET_TYPE::AIG)
        assert(0);
        // UpdAigNodeForBoolAndPartDiff(pObj);
    else if (type == NET_TYPE::SOP) {
        for (auto & pInner: cutNtk)
            UpdSopNodeForBoolAndPartDiff(pInner);
    }
    else if (type == NET_TYPE::GATE) {
        for (auto & pInner: cutNtk)
            UpdGateNodeForBoolAndPartDiff(pInner);
    }
    else
        assert(0);
    
    // get boolean difference from the node to its disjoint cuts
    bdCut2Node111.resize(disjCut.size(), dynamic_bitset <ull>(nFrame, 0));
    i = 0;
    for (auto & pCut: disjCut) {
        bdCut2Node111[i] = dat[pCut->Id] ^ tempDat[pCut->Id];
        ++i;
    }
}

bigInt GetDecOut(const boost::dynamic_bitset <ull> & poValues, ll nPo, bool isSign) {
    assert(poValues.size() == nPo);
    ll lsb = 0;
    ll msb = nPo - 1;
    ll shift = msb - lsb;
    bigInt ret(0);
    for (ll k = msb; k >= lsb; --k) {
        ret <<= 1;
        if (poValues[k])
            ++ret;
    }
    if (isSign && ret >= (bigInt(1) << shift))
        ret = -((bigInt(1) << (shift + 1)) - ret);
    return ret;
}