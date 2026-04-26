#include "my_abc.h"
#include "header.h"


using namespace abc;
using namespace std;
using namespace boost;


std::ostream & operator << (std::ostream & os, const NET_TYPE netwType) {
    const std::string strs[4] = {"AIG", "GATE", "SOP", "STRASH"};
    os << strs[static_cast <ll> (netwType)];
    return os;
}


std::ostream & operator << (std::ostream & os, const ORIENT orient) {
    const std::string strs[2] = {"AREA", "DELAY"};
    os << strs[static_cast <ll> (orient)];
    return os;
}


std::ostream & operator << (std::ostream & os, const MAP_TYPE cell) {
    const std::string strs[2] = {"LUT", "SCL"};
    os << strs[static_cast <ll> (cell)];
    return os;
}


AbcMan::AbcMan() {
    #ifdef DEBUG
    assert(Abc_FrameGetGlobalFrame() != nullptr);
    #endif
}


void AbcMan::Comm(const string & cmd, bool isVerb) {
    if (isVerb)
        cout << "Execute abc command: " << cmd << endl;
    if (Cmd_CommandExecute(GetAbcFame(), cmd.c_str())) {
        cout << "Execuation failed." << endl;
        assert(0);
    }
}


void AbcMan::ReadNet(const std::string & fileName, bool inpMapVerilog) {
    #ifdef DEBUG
    assert(IsPathExist(fileName));
    #endif
    if (inpMapVerilog)
        Comm("r -m " + fileName, true);
    else
        Comm("r " + fileName, true);
}


void AbcMan::WriteNet(const std::string & fileName, bool isVerb) {
    Comm("w " + fileName, isVerb);
}


void AbcMan::ReadStandCell(const std::string & fileName) {
    #ifdef DEBUG
    assert(IsPathExist(fileName));
    #endif
    Comm("r " + fileName);
}


void AbcMan::ConvToAig() {
    Comm("aig");
}


void AbcMan::ConvToGate() {
    Map(MAP_TYPE::SCL, ORIENT::AREA);
}


void AbcMan::ConvToSop() {
    if (GetNetType() == NET_TYPE::STRASH)
        Comm("logic;", true);
    Comm("sop", true);
}


void AbcMan::ConvToStrash() {
    Comm("st");
}


void AbcMan::PrintStat() {
    if (GetNetType() == NET_TYPE::GATE && GetAbcFame()->pLibScl != nullptr) {
        // TopoSort();
        // StatTimeAnal();
        Comm("ps");
    }
    else {
        Comm("ps");
    }
}


void AbcMan::TopoSort() {
    auto type = GetNetType();
    assert(type == NET_TYPE::AIG || type == NET_TYPE::SOP || type == NET_TYPE::GATE);
    Comm("topo");

    // fix twin nodes
    auto pNtk = GetNet();
    if (Abc_NtkHasMapping(pNtk)) {
        Abc_Ntk_t * pNtkNew; 
        Abc_Obj_t * pObj, * pFanin;
        int i, k;
        assert(pNtk != nullptr);
        // start the network
        pNtkNew = Abc_NtkStartFrom( pNtk, pNtk->ntkType, pNtk->ntkFunc );
        // copy the internal nodes
        assert(!Abc_NtkIsStrash(pNtk));
        // duplicate the nets and nodes (CIs/COs/latches already dupped)
        set <ll> skip;
        Abc_NtkForEachObj( pNtk, pObj, i ) {
            if ( pObj->pCopy == NULL && skip.count(pObj->Id) == 0 ) {
                Abc_NtkDupObj(pNtkNew, pObj, Abc_NtkHasBlackbox(pNtk) && Abc_ObjIsNet(pObj));
                auto pTwin = GetTwinNode(pObj);
                if (pTwin != nullptr) {
                    Abc_NtkDupObj(pNtkNew, pTwin, Abc_NtkHasBlackbox(pNtk) && Abc_ObjIsNet(pTwin));
                    skip.insert(pTwin->Id);
                }
            }
        }
        // reconnect all objects (no need to transfer attributes on edges)
        Abc_NtkForEachObj( pNtk, pObj, i )
            if ( !Abc_ObjIsBox(pObj) && !Abc_ObjIsBo(pObj) )
                Abc_ObjForEachFanin( pObj, pFanin, k )
                    Abc_ObjAddFanin( pObj->pCopy, pFanin->pCopy );
        // duplicate the EXDC Ntk
        if ( pNtk->pExdc )
            pNtkNew->pExdc = Abc_NtkDup( pNtk->pExdc );
        if ( pNtk->pExcare )
            pNtkNew->pExcare = Abc_NtkDup( (Abc_Ntk_t *)pNtk->pExcare );
        // duplicate timing manager
        if ( pNtk->pManTime )
            Abc_NtkTimeInitialize( pNtkNew, pNtk );
        if ( pNtk->vPhases )
            Abc_NtkTransferPhases( pNtkNew, pNtk );
        if ( pNtk->pWLoadUsed )
            pNtkNew->pWLoadUsed = Abc_UtilStrsav( pNtk->pWLoadUsed );
        // check correctness
        if ( !Abc_NtkCheck( pNtkNew ) )
            fprintf( stdout, "Abc_NtkDup(): Network check has failed.\n" );
        pNtk->pCopy = pNtkNew;
        // return pNtkNew;
        SetMainNetw(pNtkNew);
    }
}


void AbcMan::StatTimeAnal() {
    #ifdef DEBUG
    assert(GetNetType() == NET_TYPE::GATE);
    assert(GetAbcFame()->pLibScl != nullptr);
    #endif
    TopoSort();
    Comm("stime");
}


void AbcMan::Synth(ORIENT orient, bool isVerb) {
    #ifdef DEBUG
    assert(Abc_NtkIsStrash(GetNet()));
    #endif
    if (isVerb)
        cout << orient << "-oriented synthesis" << endl;
    const ll commSize = 2;
    const string areaComm[commSize] = {"st; drwsat", "st; compress2rs"};
    const string delayComm[commSize] = {"st; dc2", "st; resyn2"};
    double oldArea = GetArea();
    double oldDelay = GetDelay();
    bool isCont = true;
    while (isCont) {
        isCont = false;
        for (ll i = 0; i < commSize; ++i) {
            auto oldNtk = Abc_NtkDup(GetNet());
            if (orient == ORIENT::AREA)
                Comm(areaComm[i], isVerb);
            else if (orient == ORIENT::DELAY)
                Comm(delayComm[i], isVerb);
            else
                assert(0);
            auto res = make_pair <double, double> (GetArea(), GetDelay());
            if (isVerb)
                PrintStat();
            double newArea = res.first;
            double newDelay = res.second;
            IMPR impr = UpdNetw(oldArea, oldDelay, oldNtk, newArea, newDelay, orient);
            if (impr == IMPR::GOOD) {
                oldArea = newArea;
                oldDelay = newDelay;
                isCont = true;
            }
            if (isVerb)
                cout << (impr == IMPR::GOOD? "accept": "cancel") << endl;
        }
    }
    if (isVerb)
        PrintStat();
}


void AbcMan::SynthWithResyn2Comm() {
    Comm("st; resyn2; logic; sop; ps;");
}


void AbcMan::SynthAIG() {
    Comm("st; logic; sop; ps;");
}


bool AbcMan::SynthAndMap(double maxDelay, bool isVerb) {
    bool fGenNewSubNodes = false;
    bool cont = true;
    if (isVerb)
        cout << "Begin SynthAndMap: maxDelay = " << maxDelay << endl;
    TopoSort();
    while (cont) {
        double oldArea = numeric_limits <double>::max(), oldDelay = numeric_limits <double>::max();
        if (GetNetType(GetNet()) == NET_TYPE::GATE)
            oldArea = GetArea(), oldDelay = GetDelay();
        auto pOldNtk = Abc_NtkDup(GetNet());
        if (isVerb)
            cout << "oldArea = " << oldArea << ", " << "oldDelay = " << oldDelay << endl;
        // Comm("st; resyn2; dch; amap;", isVerb);
        Comm("st; resyn; dch; amap;", isVerb);
        TopoSort();
        double newArea = GetArea(), newDelay = GetDelay();
        if (isVerb)
            cout << "newArea = " << newArea << ", " << "newDelay = " << newDelay << endl;
        if (newDelay <= maxDelay) {
            auto impr = UpdNetw(oldArea, oldDelay, pOldNtk, newArea, newDelay, ORIENT::AREA);
            if (impr != IMPR::GOOD) {
                cont = false;
                if (isVerb)
                    cout << "reject" << endl;
            }
            else {
                if (isVerb)
                    cout << "accept" << endl;
                if (!fGenNewSubNodes)
                    fGenNewSubNodes = true;
            }
        }
        else {
            SetMainNetw(pOldNtk);
                cont = false;
            if (isVerb)
                cout << "reject" << endl;
        }
    }
    PrintStat();
    return fGenNewSubNodes;
}

bool AbcMan::SynthAndMap_v2(double maxDelay, bool isVerb, ll selectComm) {
    bool fGenNewSubNodes = false;
    bool cont = true;
    if (isVerb)
        cout << "Begin SynthAndMap: maxDelay = " << maxDelay << ", selectComm = " << selectComm << endl;
    TopoSort();
    while (cont) {
        double oldArea = numeric_limits <double>::max(), oldDelay = numeric_limits <double>::max();
        if (GetNetType(GetNet()) == NET_TYPE::GATE)
            oldArea = GetArea(), oldDelay = GetDelay();
        auto pOldNtk = Abc_NtkDup(GetNet());
        if (isVerb)
            cout << "oldArea = " << oldArea << ", " << "oldDelay = " << oldDelay << endl;
        // Comm("st; resyn2; dch; amap;", isVerb);
        if (selectComm == 1) {
            Comm("st; resyn; dch; amap;", isVerb);
        }
        else if (selectComm == 2) {
            ostringstream oss("");
            oss << "st; resyn; map -D " << maxDelay;
            Comm(oss.str(), isVerb);
        }
        else if (selectComm == 3) {
            ostringstream oss("");
            oss << "st; resyn2rs; map -D " << maxDelay;
            Comm(oss.str(), isVerb);
        }
        else if (selectComm == 4) {
            ostringstream oss("");
            oss << "st; compress2rs; map -D " << maxDelay;
            Comm(oss.str(), isVerb);
        }
        else if (selectComm == 5) {
            Comm("st; resyn2rs; map -a;", isVerb);
        }
        else if (selectComm == 6) {
            Comm("st; resyn2; map -a;", isVerb);
        }
        else if (selectComm == 7) {
            Comm("st; resyn2; resyn2; map -a;", isVerb);
        }
        else if (selectComm == 8) {
            Comm("st; resyn2; resyn2; dch; map -a;", isVerb);
        }
        else if (selectComm == 9) {
            ostringstream oss("");
            oss << "st; compress2rs; map -D " << maxDelay << " -a;";
            Comm(oss.str(), isVerb);
        }
        else if (selectComm == 10) {
            ostringstream oss("");
            oss << "st; compress2rs; resyn2; map -D " << maxDelay << " -a;";
            Comm(oss.str(), isVerb);
        }
        else if (selectComm == 11) {
            Comm("st; resyn2; amap;", isVerb);
        }
        else if (selectComm == 12) {
            Comm("st; dc2; dch; resyn2; amap;", isVerb);
        }
        else if (selectComm == 13) {
            Comm("st; compress2rs; resyn2; amap;", isVerb);
        }
        else if (selectComm == 14) {
            Comm("st; resyn2; resyn2; balance; compress2rs; amap;", isVerb);
        }
        else
            assert(0);
        TopoSort();
        double newArea = GetArea(), newDelay = GetDelay();
        if (isVerb)
            cout << "newArea = " << newArea << ", " << "newDelay = " << newDelay << endl;
        if (newDelay <= maxDelay) {
            auto impr = UpdNetw(oldArea, oldDelay, pOldNtk, newArea, newDelay, ORIENT::AREA);
            if (impr != IMPR::GOOD) {
                cont = false;
                if (isVerb)
                    cout << "reject" << endl;
            }
            else {
                if (isVerb)
                    cout << "accept" << endl;
                if (!fGenNewSubNodes)
                    fGenNewSubNodes = true;
            }
        }
        else {
            SetMainNetw(pOldNtk);
                cont = false;
            if (isVerb)
                cout << "reject" << endl;
        }
    }
    PrintStat();
    return fGenNewSubNodes;
}


// void AbcMan::SynthAndMap3(bool isVerb) {
//     #ifdef DEBUG
//     cout << "simple resyn2-based synthesis & mapping for SCL" << endl;
//     #endif
//     ostringstream comm;
//     // comm << "st; resyn2; st; dch; map -D " << maxDelay;
//     comm << "st; resyn2; st; dch; map";
//     Comm(comm.str());
//     if (isVerb)
//         PrintStat();
// }


void AbcMan::Sweep() {
    #ifdef DEBUG
    assert(GetNetType() == NET_TYPE::SOP);
    #endif
    Comm("sweep; sop;");
    // Comm("sweep -s; map");
}


pair <double, double> AbcMan::Map(MAP_TYPE cell, ORIENT orient, bool isVerb) {
    double oldArea = numeric_limits <double>::max();
    double oldDelay = numeric_limits <double>::max();
    ostringstream LutInpStr("");
    LutInpStr << LutInp;
    if ((cell == MAP_TYPE::SCL && GetNetType() == NET_TYPE::GATE) ||
        (cell == MAP_TYPE::LUT && IsLutNetw())) {
        oldArea = GetArea();
        oldDelay = GetDelay();
    }
    bool isFirst = true;
    bool isCont = true;
    while (isCont) {
        auto oldNtk = Abc_NtkDup(GetNet());
        if (isFirst) {
            Comm("st; dch;", isVerb);
            isFirst = false;
        }
        else
            Comm("st; b;", isVerb);
        if (cell == MAP_TYPE::SCL) {
            if (orient == ORIENT::AREA)
                Comm("amap", isVerb);
            else if (orient == ORIENT::DELAY)
                Comm("map", isVerb);
            else
                assert(0);
        }
        else if (cell == MAP_TYPE::LUT) {
            if (orient == ORIENT::AREA)
                Comm("if -a -K " + LutInpStr.str(), isVerb);
            else if (orient == ORIENT::DELAY)
                Comm("if -K " + LutInpStr.str(), isVerb);
            else
                assert(0);
        }
        else
            assert(0);
        double newArea = GetArea();
        double newDelay = GetDelay();
        IMPR impr = UpdNetw(oldArea, oldDelay, oldNtk, newArea, newDelay, orient);
        if (impr == IMPR::GOOD) {
            oldArea = newArea;
            oldDelay = newDelay;
        }
        else
            isCont = false;
        // PrintStat();
    }
    return make_pair(oldArea, oldDelay);
}


pair <double, double> AbcMan::Map2(double maxDelay, bool isVerb) {
    double oldArea = numeric_limits <double>::max();
    double oldDelay = numeric_limits <double>::max();
    ostringstream LutInpStr("");
    LutInpStr << LutInp;
    assert(GetNetType() == NET_TYPE::STRASH);
    bool isFirst = true;
    bool isCont = true;
    while (isCont) {
        auto oldNtk = Abc_NtkDup(GetNet());
        if (isFirst) {
            Comm("st; dch;", isVerb);
            isFirst = false;
        }
        else
            Comm("st; b;", isVerb);
        ostringstream oss("");
        oss << "map -D " << maxDelay;
        Comm(oss.str(), isVerb);
        double newArea = GetArea();
        double newDelay = GetDelay();
        IMPR impr = UpdNetw(oldArea, oldDelay, oldNtk, newArea, newDelay, ORIENT::AREA);
        if (impr == IMPR::GOOD) {
            oldArea = newArea;
            oldDelay = newDelay;
        }
        else
            isCont = false;
        // PrintStat();
    }
    return make_pair(oldArea, oldDelay);
}


IMPR AbcMan::UpdNetw(double oldArea, double oldDelay, Abc_Ntk_t * oldNtk, double newArea, double newDelay, ORIENT orient) {
    IMPR impr = IMPR::SAME;
    if (orient == ORIENT::AREA) {
        // if (DoubleGreat(newArea, oldArea) || (DoubleEqual(newArea, oldArea) && DoubleGreat(newDelay, oldDelay)))
        if (DoubleGreat(newArea, oldArea) || DoubleGreat(newDelay, oldDelay))
            impr = IMPR::BAD;
        else if (DoubleEqual(newArea, oldArea) && DoubleEqual(newDelay, oldDelay))
            impr = IMPR::SAME;
        else
            impr = IMPR::GOOD;
    }
    else if (orient == ORIENT::DELAY) {
        if (DoubleGreat(newDelay, oldDelay) || (DoubleEqual(newDelay, oldDelay) && DoubleGreat(newArea, oldArea)))
            impr = IMPR::BAD;
        else if (DoubleEqual(newDelay, oldDelay) && DoubleEqual(newArea, oldArea))
            impr = IMPR::SAME;
        else
            impr = IMPR::GOOD;
    }
    else
        assert(0);
    if (impr == IMPR::BAD) {
        #ifdef DEBUG
        assert(oldArea != numeric_limits <double>::max() && oldDelay != numeric_limits <double>::max());
        assert(oldNtk != nullptr);
        // cout << "Cancel the last abc command" << endl;
        #endif
        SetMainNetw(oldNtk);
    }
    else
        Abc_NtkDelete(oldNtk);
    return impr;
}


NET_TYPE AbcMan::GetNetType(Abc_Ntk_t * pNtk) const {
    if (Abc_NtkIsAigLogic(pNtk))
        return NET_TYPE::AIG;
    else if (Abc_NtkIsMappedLogic(pNtk))
        return NET_TYPE::GATE;
    else if (Abc_NtkIsSopLogic(pNtk))
        return NET_TYPE::SOP;
    else if (Abc_NtkIsStrash(pNtk))
        return NET_TYPE::STRASH;
    else {
        cout << pNtk << endl;
        cout << "invalid network type" << endl;
        assert(0);
    }
}


double AbcMan::GetArea(Abc_Ntk_t * pNtk) const {
    auto type = GetNetType(pNtk);
    if (type == NET_TYPE::AIG || type == NET_TYPE::STRASH)
        return Abc_NtkNodeNum(pNtk);
    else if (type == NET_TYPE::SOP) {
        Abc_Obj_t * pObj = nullptr;
        ll i = 0;
        ll ret = Abc_NtkNodeNum(pNtk);
        Abc_NtkForEachNode(pNtk, pObj, i) {
            if (Abc_NodeIsConst(pObj))
                --ret;
        }
        return ret;
    }
    else if (type == NET_TYPE::GATE) {
        auto pLibScl = static_cast <SC_Lib *> (GetAbcFame()->pLibScl);
        if (pLibScl == nullptr)
            return Abc_NtkGetMappedArea(pNtk);
        else {
            return Abc_NtkGetMappedArea(pNtk);
            // #ifdef DEBUG
            // assert(pNtk->nBarBufs2 == 0);
            // assert(CheckSCLNet(pNtk));
            // #endif
            // SC_Man * p = Abc_SclManStart(pLibScl, pNtk, 0, 1, 0, 0);
            // double area = Abc_SclGetTotalArea(p->pNtk);
            // Abc_SclManFree(p);
            // return area;
        }
    }
    else
        assert(0);
}


double AbcMan::GetDelay(Abc_Ntk_t * pNtk) const {
    auto type = GetNetType(pNtk);
    if (type == NET_TYPE::AIG || type == NET_TYPE::SOP || type == NET_TYPE::STRASH)
        return Abc_NtkLevel(pNtk);
    else if (type == NET_TYPE::GATE) {
        auto pLibScl = static_cast <SC_Lib *> (GetAbcFame()->pLibScl);
        if (pLibScl == nullptr) {
            return Abc_NtkDelayTrace(pNtk, nullptr, nullptr, 0);
        }
        else {
            #ifdef DEBUG
            assert(pNtk->nBarBufs2 == 0);
            assert(CheckSCLNet(pNtk));
            #endif
            SC_Man * p = Abc_SclManStart(pLibScl, pNtk, 0, 1, 0, 0);
            int fRise = 0;
            Abc_Obj_t * pPivot = Abc_SclFindCriticalCo(p, &fRise); 
            double delay = Abc_SclObjTimeOne(p, pPivot, fRise);
            Abc_Obj_t * pObj = nullptr;
            ll i = 0;
            Abc_NtkForEachObj(pNtk, pObj, i)
                pObj->dTemp = Abc_SclObjTimeMax(p, pObj);
            Abc_SclManFree(p);
            return delay;
        }
    }
    else
        assert(0);
}

bool CheckSCLNet(abc::Abc_Ntk_t * pNtk) {
    Abc_Obj_t * pObj, * pFanin;
    int i, k, fFlag = 1;
    Abc_NtkIncrementTravId( pNtk );        
    Abc_NtkForEachCi( pNtk, pObj, i )
        Abc_NodeSetTravIdCurrent( pObj );
    Abc_NtkForEachNode( pNtk, pObj, i )
    {
        Abc_ObjForEachFanin( pObj, pFanin, k )
            if ( !Abc_NodeIsTravIdCurrent( pFanin ) )
                printf( "obj %d and its fanin %d are not in the topo order\n", Abc_ObjId(pObj), Abc_ObjId(pFanin) ), fFlag = 0;
        Abc_NodeSetTravIdCurrent( pObj );
        if ( Abc_ObjIsBarBuf(pObj) )
            continue;
        if ( !fFlag )
            break;
    }
    return fFlag;
}

double GetDelay(abc::Abc_Ntk_t * pNtk) {
    assert(Abc_NtkIsMappedLogic(pNtk));
    assert(pNtk->nBarBufs2 == 0);
    assert(CheckSCLNet(pNtk));

    auto pLibScl = static_cast <SC_Lib *> (Abc_FrameGetGlobalFrame()->pLibScl);
    SC_Man * p = Abc_SclManStart(pLibScl, pNtk, 0, 1, 0, 0);
    int fRise = 0;
    Abc_Obj_t * pPivot = Abc_SclFindCriticalCo(p, &fRise); 
    double delay = Abc_SclObjTimeOne(p, pPivot, fRise);
    Abc_Obj_t * pObj = nullptr;
    ll i = 0;
    Abc_NtkForEachObj(pNtk, pObj, i)
        pObj->dTemp = Abc_SclObjTimeMax(p, pObj);
    Abc_SclManFree(p);
    return delay;
}


bool AbcMan::CheckSCLNet(abc::Abc_Ntk_t * pNtk) const {
    Abc_Obj_t * pObj, * pFanin;
    int i, k, fFlag = 1;
    Abc_NtkIncrementTravId( pNtk );        
    Abc_NtkForEachCi( pNtk, pObj, i )
        Abc_NodeSetTravIdCurrent( pObj );
    Abc_NtkForEachNode( pNtk, pObj, i )
    {
        Abc_ObjForEachFanin( pObj, pFanin, k )
            if ( !Abc_NodeIsTravIdCurrent( pFanin ) )
                printf( "obj %d and its fanin %d are not in the topo order\n", Abc_ObjId(pObj), Abc_ObjId(pFanin) ), fFlag = 0;
        Abc_NodeSetTravIdCurrent( pObj );
        if ( Abc_ObjIsBarBuf(pObj) )
            continue;
        // if ( Abc_ObjFanoutNum(pObj) == 0 )
        //     printf( "node %d has no fanout\n", Abc_ObjId(pObj) ), fFlag = 0;
        if ( !fFlag )
            break;
    }
    // if ( fFlag && fVerbose )
    //     printf( "The network is in topo order and no dangling nodes.\n" );
    return fFlag;
}

vector <abc::Abc_Obj_t * > AbcMan::GetIdentNode( Abc_Obj_t * pNode ) {
    assert( Abc_NtkHasMapping(pNode->pNtk) );
    Mio_Gate_t * pGate = (Mio_Gate_t *)pNode->pData;
    vector <abc::Abc_Obj_t * > pIdent;
    if ( pGate == nullptr || Mio_GateReadTwin(pGate) == nullptr )
        return pIdent;
    Abc_Obj_t * pNode2 = nullptr;
    ll id = 0;
    Abc_NtkForEachNode(pNode->pNtk, pNode2, id) {
        if (pNode == pNode2)
            continue;
        if ( Abc_ObjFaninNum(pNode) != Abc_ObjFaninNum(pNode2) )
            continue;
        bool sameFanin = true;
        for (ll faninId = 0; faninId < Abc_ObjFaninNum(pNode); ++faninId) {
            if (Abc_ObjFanin(pNode, faninId) != Abc_ObjFanin(pNode2, faninId)) {
                sameFanin = false;
                break;
            }
        }
        if (!sameFanin)
            continue;
        Mio_Gate_t * pGate2 = (Mio_Gate_t *)pNode2->pData;
        if ( Mio_GateReadName(pGate) != Mio_GateReadName(pGate2))
            continue;
        if ( Mio_GateReadTwin(pGate) == (Mio_Gate_t *)pNode2->pData )
            continue;
        pIdent.emplace_back(pNode2);
    }
    return pIdent;
}


Abc_Obj_t * AbcMan::GetTwinNode( Abc_Obj_t * pNode ) {
    assert( Abc_NtkHasMapping(pNode->pNtk) );
    Mio_Gate_t * pGate = (Mio_Gate_t *)pNode->pData;
    if ( pGate == nullptr || Mio_GateReadTwin(pGate) == nullptr )
        return nullptr;
    Abc_Obj_t * pNode2 = nullptr;
    ll id = 0;
    Abc_Obj_t * pTwin = nullptr;
    ll count = 0;
    Abc_NtkForEachNode(pNode->pNtk, pNode2, id) {
        if ( Abc_ObjFaninNum(pNode) != Abc_ObjFaninNum(pNode2) )
            continue;
        bool sameFanin = true;
        for (ll faninId = 0; faninId < Abc_ObjFaninNum(pNode); ++faninId) {
            if (Abc_ObjFanin(pNode, faninId) != Abc_ObjFanin(pNode2, faninId)) {
                sameFanin = false;
                break;
            }
        }
        if (!sameFanin)
            continue;
        if ( Mio_GateReadTwin(pGate) != (Mio_Gate_t *)pNode2->pData )
            continue;
        pTwin = pNode2;
        // return pTwin;
        ++count;
        if (count > 1){
            cout << "the second twin node is " << Abc_ObjName(pTwin) << endl;
            cout << "the target node is " << Abc_ObjName(pNode) << endl;
            cout << "the target node has " << Abc_ObjName(pNode)  << " fanins. "<< endl;
            auto ident = GetIdentNode(pTwin);
            cout << "The identical node(s) of " << Abc_ObjName(pTwin) << ": ";
            for (auto & k : ident)
                cout << Abc_ObjName(k) << " ";
            cout << endl;
            assert(0);
        }
    }
    return pTwin;
}


void AbcMan::LoadAlias() {
    Comm("alias hi history", false);
    Comm("alias b balance", false);
    Comm("alias cg clockgate", false);
    Comm("alias cl cleanup", false);
    Comm("alias clp collapse", false);
    Comm("alias cs care_set", false);
    Comm("alias el eliminate", false);
    Comm("alias esd ext_seq_dcs", false);
    Comm("alias f fraig", false);
    Comm("alias fs fraig_sweep", false);
    Comm("alias fsto fraig_store", false);
    Comm("alias fres fraig_restore", false);
    Comm("alias fr fretime", false);
    Comm("alias ft fraig_trust", false);
    Comm("alias ic indcut", false);
    Comm("alias lp lutpack", false);
    Comm("alias pcon print_cone", false);
    Comm("alias pd print_dsd", false);
    Comm("alias pex print_exdc -d", false);
    Comm("alias pf print_factor", false);
    Comm("alias pfan print_fanio", false);
    Comm("alias pg print_gates", false);
    Comm("alias pl print_level", false);
    Comm("alias plat print_latch", false);
    Comm("alias pio print_io", false);
    Comm("alias pk print_kmap", false);
    Comm("alias pm print_miter", false);
    Comm("alias ps print_stats ", false);
    Comm("alias psb print_stats -b", false);
    Comm("alias psu print_supp", false);
    Comm("alias psy print_symm", false);
    Comm("alias pun print_unate", false);
    Comm("alias q quit", false);
    Comm("alias r read", false);
    Comm("alias ra read_aiger", false);
    Comm("alias r3 retime -M 3", false);
    Comm("alias r3f retime -M 3 -f", false);
    Comm("alias r3b retime -M 3 -b", false);
    Comm("alias ren renode", false);
    Comm("alias rh read_hie", false);
    Comm("alias ri read_init", false);
    Comm("alias rl read_blif", false);
    Comm("alias rb read_bench", false);
    Comm("alias ret retime", false);
    Comm("alias dret dretime", false);
    Comm("alias rp read_pla", false);
    Comm("alias rt read_truth", false);
    Comm("alias rv read_verilog", false);
    Comm("alias rvl read_verlib", false);
    Comm("alias rsup read_super mcnc5_old.super", false);
    Comm("alias rlib read_library", false);
    Comm("alias rlibc read_library cadence.genlib", false);
    Comm("alias rty read_liberty", false);
    Comm("alias rlut read_lut", false);
    Comm("alias rw rewrite", false);
    Comm("alias rwz rewrite -z", false);
    Comm("alias rf refactor", false);
    Comm("alias rfz refactor -z", false);
    Comm("alias re restructure", false);
    Comm("alias rez restructure -z", false);
    Comm("alias rs resub", false);
    Comm("alias rsz resub -z", false);
    Comm("alias sa set autoexec ps", false);
    Comm("alias scl scleanup", false);
    Comm("alias sif if -s", false);
    Comm("alias so source -x", false);
    Comm("alias st strash", false);
    Comm("alias sw sweep", false);
    Comm("alias ssw ssweep", false);
    Comm("alias tr0 trace_start", false);
    Comm("alias tr1 trace_check", false);
    Comm("alias trt \"r c.blif; st; tr0; b; tr1\"", false);
    Comm("alias u undo", false);
    Comm("alias w write", false);
    Comm("alias wa write_aiger", false);
    Comm("alias wb write_bench", false);
    Comm("alias wc write_cnf", false);
    Comm("alias wh write_hie", false);
    Comm("alias wl write_blif", false);
    Comm("alias wp write_pla", false);
    Comm("alias wv write_verilog", false);
    Comm("alias resyn       \"b; rw; rwz; b; rwz; b\"", false);
    Comm("alias resyn2      \"b; rw; rf; b; rw; rwz; b; rfz; rwz; b\"", false);
    Comm("alias resyn2a     \"b; rw; b; rw; rwz; b; rwz; b\"", false);
    Comm("alias resyn3      \"b; rs; rs -K 6; b; rsz; rsz -K 6; b; rsz -K 5; b\"", false);
    Comm("alias compress    \"b -l; rw -l; rwz -l; b -l; rwz -l; b -l\"", false);
    Comm("alias compress2   \"b -l; rw -l; rf -l; b -l; rw -l; rwz -l; b -l; rfz -l; rwz -l; b -l\"", false);
    Comm("alias choice      \"fraig_store; resyn; fraig_store; resyn2; fraig_store; fraig_restore\"", false);
    Comm("alias choice2     \"fraig_store; balance; fraig_store; resyn; fraig_store; resyn2; fraig_store; resyn2; fraig_store; fraig_restore\"", false);
    Comm("alias rwsat       \"st; rw -l; b -l; rw -l; rf -l\"", false);
    Comm("alias drwsat2     \"st; drw; b -l; drw; drf; ifraig -C 20; drw; b -l; drw; drf\"", false);
    Comm("alias share       \"st; multi -m; sop; fx; resyn2\"", false);
    Comm("alias addinit     \"read_init; undc; strash; zero\"", false);
    Comm("alias blif2aig    \"undc; strash; zero\"", false);
    Comm("alias v2p         \"&vta_gla; &ps; &gla_derive; &put; w 1.aig; pdr -v\"", false);
    Comm("alias g2p         \"&ps; &gla_derive; &put; w 2.aig; pdr -v\"", false);
    Comm("alias &sw_        \"&put; sweep; st; &get\"", false);
    Comm("alias &fx_        \"&put; sweep; sop; fx; st; &get\"", false);
    Comm("alias &dc3        \"&b; &jf -K 6; &b; &jf -K 4; &b\"", false);
    Comm("alias &dc4        \"&b; &jf -K 7; &fx; &b; &jf -K 5; &fx; &b\"", false);
    Comm("alias src_rw      \"st; rw -l; rwz -l; rwz -l\"", false);
    Comm("alias src_rs      \"st; rs -K 6 -N 2 -l; rs -K 9 -N 2 -l; rs -K 12 -N 2 -l\"", false);
    Comm("alias src_rws     \"st; rw -l; rs -K 6 -N 2 -l; rwz -l; rs -K 9 -N 2 -l; rwz -l; rs -K 12 -N 2 -l\"", false);
    Comm("alias resyn2rs    \"b; rs -K 6; rw; rs -K 6 -N 2; rf; rs -K 8; b; rs -K 8 -N 2; rw; rs -K 10; rwz; rs -K 10 -N 2; b; rs -K 12; rfz; rs -K 12 -N 2; rwz; b\"", false);
    Comm("alias compress2rs \"b -l; rs -K 6 -l; rw -l; rs -K 6 -N 2 -l; rf -l; rs -K 8 -l; b -l; rs -K 8 -N 2 -l; rw -l; rs -K 10 -l; rwz -l; rs -K 10 -N 2 -l; b -l; rs -K 12 -l; rfz -l; rs -K 12 -N 2 -l; rwz -l; b -l\"", false);
    Comm("alias fix_aig     \"logic; undc; strash; zero\"", false);
    Comm("alias fix_blif    \"undc; strash; zero\"", false);
    Comm("alias recadd3     \"st; rec_add3; b; rec_add3; dc2; rec_add3; if -K 8; bidec; st; rec_add3; dc2; rec_add3; if -g -K 6; st; rec_add3\"", false);
}


NetMan::NetMan(): AbcMan(), pNtk(nullptr), isDupl(true), level(0) {
    // cout << "construct empty network" << endl;
}


NetMan::NetMan(Abc_Ntk_t * p_ntk, bool is_dupl): AbcMan(), isDupl(is_dupl), level(0) {
    // cout << "construct netman" << endl;
    // cout << "old network = " << p_ntk << endl;
    if (is_dupl)
        pNtk = Abc_NtkDup(p_ntk);
    else
        pNtk = p_ntk;
    // cout << "new network = " << pNtk << endl;
}


NetMan::NetMan(std::string & fileName): AbcMan() {
    AbcMan::ReadNet(fileName);
    pNtk = AbcMan::GetNet();
}


NetMan::~NetMan() {
    // cout << "destroy network" << endl;
    if (isDupl && pNtk != AbcMan::GetNet()) {
        if (pNtk != nullptr) {
            // cout << "delete duplicated network " << pNtk << endl;
            // cout << "frame network " << AbcMan::GetNet() << endl;
            Abc_NtkDelete(pNtk);
            pNtk = nullptr;
        }
    }
}


NetMan::NetMan(const NetMan & net_man): AbcMan(), isDupl(true) {
    // cout << "copy netman" << endl;
    pNtk = Abc_NtkDup(net_man.pNtk);
    level = net_man.level;
}


NetMan::NetMan(NetMan && net_man): AbcMan(), pNtk(net_man.pNtk), isDupl(net_man.isDupl), level(net_man.level) {
    // cout << "move copy netman" << endl;
    net_man.isDupl = false;
    net_man.pNtk = nullptr;
    net_man.level = 0;
}


NetMan & NetMan::operator = (const NetMan & net_man) {
    // cout << "assign netman" << endl;
    if (this == &net_man)
        return *this;
    if (isDupl && pNtk != nullptr && pNtk != AbcMan::GetNet() && pNtk != net_man.GetNet())
        Abc_NtkDelete(pNtk);
    pNtk = Abc_NtkDup(net_man.GetNet());
    isDupl = true;
    level = net_man.GetLevel();
    return *this;
}


NetMan & NetMan::operator = (NetMan && net_man) {
    // cout << "move assign netman" << endl;
    if (this == &net_man)
        return *this;
    if (isDupl && pNtk != nullptr && pNtk != AbcMan::GetNet() && pNtk != net_man.GetNet())
        Abc_NtkDelete(pNtk);
    pNtk = net_man.pNtk;
    isDupl = net_man.isDupl;
    level = net_man.level;
    net_man.isDupl = false;
    net_man.pNtk = nullptr;
    net_man.level = 0;
    return *this;
}


pair <ll, ll> NetMan::GetConstId(bool isVerb) {
    pair <ll, ll> ret(-1, -1);
    auto type = GetNetType();
    Abc_Obj_t * pObj = nullptr;
    ll i = 0;
    Abc_NtkForEachNode(GetNet(), pObj, i) {
        if (type == NET_TYPE::GATE || type == NET_TYPE::SOP) {
            if (Abc_NodeIsConst0(pObj)) {
                if (isVerb)
                    cout << "find const 0: " << pObj << endl;
                if (ret.first == -1)
                    ret.first = GetId(pObj);
            }
            else if (Abc_NodeIsConst1(pObj)) {
                if (isVerb)
                    cout << "find const 1: " << pObj << endl;
                if (ret.second == -1)
                    ret.second = GetId(pObj);
            }
        }
        else if (type == NET_TYPE::AIG) {
            auto pHopObj = static_cast <Hop_Obj_t *> (pObj->pData);
            auto pHopObjR = Hop_Regular(pHopObj);
            if (Hop_ObjIsConst1(pHopObjR)) {
                #ifdef DEBUG
                assert(Hop_ObjFanin0(pHopObjR) == nullptr);
                assert(Hop_ObjFanin1(pHopObjR) == nullptr);
                #endif
                if (!Hop_IsComplement(pHopObj))
                    ret.second = GetId(pObj);
                else 
                    ret.first = GetId(pObj);
            }
        }
        else
            assert(0);
    }
    return ret;
}


pair <ll, ll> NetMan::CreateConst(bool isVerb) {
    auto consts = GetConstId(isVerb);
    pair <ll, ll> ret(consts);
    if (ret.first == -1) {
        auto pObj = Abc_NtkCreateNodeConst0(GetNet());
        RenameAbcObj(pObj, "const0");
        ret.first = GetId(pObj);
        if (isVerb)
            cout << "create const 0: " << pObj << endl;
    }
    if (ret.second == -1) {
        auto pObj = Abc_NtkCreateNodeConst1(GetNet());
        RenameAbcObj(pObj, "const1");
        ret.second = GetId(pObj);
        if (isVerb)
            cout << "create const 1: " << pObj << endl;
    }
    return ret;
}

ll NetMan::CreateOneConst(bool isConst0) {
    Abc_Obj_t * pObj;
    if (isConst0)
        pObj = Abc_NtkCreateNodeConst0(GetNet());
    else
        pObj = Abc_NtkCreateNodeConst1(GetNet());
    return GetId(pObj);

}

// ll NetMan::GetConstNum(bool isConst0)
void NetMan::PrintNodeLev(std::vector <ll> & targIds){
    vector <ll> LevList(60,0);
    ll count = 0;
    GetLev();
    for (auto & targId : targIds){
        auto tmplev = GetObjLev(targId);
        LevList[tmplev] += 1;
        count++;
    }
    for (auto i = 0; i < LevList.size(); ++i){
        // cout << "Lev " << i << ": ";
        cout << LevList[i] << endl;
    }
    cout << "Node counting: " << count << endl;
}

void NetMan::CreateConstNodes(bool isVerb) {
    auto type = GetNetType();
    assert(type == NET_TYPE::GATE || type == NET_TYPE::SOP);
    auto consts = CreateConst();
    auto pObj = GetObj(consts.first);
    vector <Abc_Obj_t *> fanouts0;
    for (ll i = 0; i < GetFanoutNum(pObj); ++i) {
        auto fanout = GetFanout(pObj, i);
        if (IsObjPo(fanout))
            continue;
        fanouts0.emplace_back(fanout);
    }
    for (auto & fanout : fanouts0) {
        for (ll j = 0; j < GetFaninNum(fanout); ++j){
            auto pDriv = GetFanin(fanout, j);
            if (pDriv == pObj){
                auto pConst0 = Abc_NtkCreateNodeConst0(GetNet());
                Abc_ObjPatchFanin(fanout, pDriv, pConst0);
            }
        }
    }
    pObj = GetObj(consts.second);
    vector <Abc_Obj_t *> fanouts1;
    for (ll i = 0; i < GetFanoutNum(pObj); ++i) {
        auto fanout = GetFanout(pObj, i);
        if (IsObjPo(fanout))
            continue;
        fanouts1.emplace_back(fanout);
    }
    for (auto & fanout : fanouts1) {
        for (ll j = 0; j < GetFaninNum(fanout); ++j){
            auto pDriv = GetFanin(fanout, j);
            if (pDriv == pObj){
                auto pConst1 = Abc_NtkCreateNodeConst1(GetNet());
                Abc_ObjPatchFanin(fanout, pDriv, pConst1);
            }
        }
    }
    CleanUp();
}


vector <ll> NetMan::FindPartialPro(){
    auto type = GetNetType();
    assert(type == NET_TYPE::GATE);
    Abc_Obj_t * pObj = nullptr;
    int i = 0;
    vector <ll> TarNodes;
    Abc_NtkForEachNode(GetNet(), pObj, i) {
        for (auto j = 0; j < GetFanoutNum(pObj); ++j){
            auto pFanout = GetFanout(pObj, j);
            if (GetGateName(pFanout).find("HA1") == -1 || GetGateName(pFanout).find("FA1") == -1){
                TarNodes.emplace_back(GetId(pObj));
                break;
            }
        }
        // TarNodes.emplace_back(GetId(pObj));
    }
    return TarNodes;
}

void NetMan::AddInv(ll & targId){
    // auto type = GetNetType();
    // assert(type == NET_TYPE::GATE);
    // Abc_Obj_t * pObj = GetObj(targId);
    // auto pTwin = GetTwinNode(pObj);
    // auto twinId = GetId(pTwin);
    // auto pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targId, ifanin));
    // assert(GetFanin(targId, ifanin) == GetFanin(twinId, ifanin));
    // PatchFanin(pObj, ifanin, GetFanin(targId, ifanin), pSub);
    // PatchFanin(pTwin, ifanin, GetFanin(twinId, ifanin), pSub);
    auto type = GetNetType();
    assert(type == NET_TYPE::GATE);
    Abc_Obj_t * pObj = GetObj(targId);
    auto pSub = Abc_NtkCreateNodeInv(GetNet(), pObj);
    for (auto i = 0; i < GetFanoutNum(pObj); ++i){
        auto pFanout = GetFanout(pObj,i);
        for (auto j = 0; j < GetFaninNum(pFanout); ++j){
            if (GetFanin(pFanout, j) == pObj)
                PatchFanin(pFanout, j, pObj, pSub);
        }
    }
}
void NetMan::ApplyCompensation(int RealCom, ll truncBit){
    if (GetNetType() != NET_TYPE::GATE)
        return;
    auto Com = RealCom >> truncBit;
    vector <int> ComList;
    for (auto bit = truncBit; bit < GetPoNum(); ++bit){
        ComList.emplace_back(Com % 2);
        Com = Com >> 1;
    }
    for (auto & com : ComList){
        cout << com << " ";
    }
    cout << endl;
    Abc_Obj_t * tmpCO = nullptr;
    Abc_Obj_t * pCO = nullptr;
    Abc_Obj_t * pS = nullptr;
    for (auto i = truncBit; i < GetPoNum(); ++i){
        auto pPo = GetPo(i);
        assert(GetFaninNum(pPo) == 1);
        auto pDrive = GetFanin(pPo, 0);
        // cout << pPo << " with fanin " << pDrive << endl;
        auto consts = CreateConst();
        Abc_Obj_t * ConstNode = nullptr;
        if (ComList[i - truncBit] == 0)
            ConstNode = GetObj(consts.first);
        else if (ComList[i - truncBit] == 1)
            ConstNode = GetObj(consts.second);
        else
            assert(0);
        WriteNet("test1.v");
        auto pNewNode = Abc_NtkCreateNode(GetNet());
        for (ll faninId = 0; faninId < GetFaninNum(pDrive); ++faninId)
            Abc_ObjAddFanin(pNewNode, GetFanin(pDrive, faninId));
        Mio_Gate_t * pGate = (Mio_Gate_t *)pDrive->pData;
        pNewNode->pData = pGate;
        if (i == truncBit){
            pCO = CreateGate2(std::vector <Abc_Obj_t *> ({pNewNode, ConstNode}), "HA1D0BWP7T30P140HVT", "CO");
            pS = CreateGate2(std::vector <Abc_Obj_t *> ({pNewNode, ConstNode}), "HA1D0BWP7T30P140HVT", "S");
            TransfFanout(pDrive, pS);
            tmpCO = pCO;
        }
        else if (i == GetPoNum() - 1){
            pS = CreateGate2(std::vector <Abc_Obj_t *> ({pNewNode, tmpCO, ConstNode}), "FA1D0BWP7T30P140HVT", "S");
            TransfFanout(pDrive, pS);
        }
        else {
            pCO = CreateGate2(std::vector <Abc_Obj_t *> ({pNewNode, tmpCO, ConstNode}), "FA1D0BWP7T30P140HVT", "CO");
            pS = CreateGate2(std::vector <Abc_Obj_t *> ({pNewNode, tmpCO, ConstNode}), "FA1D0BWP7T30P140HVT", "S");
            TransfFanout(pDrive, pS);
            tmpCO = pCO;
        }
    }
    WriteNet("test2.v");
    CleanUp();
}

void NetMan::ProcVerTrun(vector <pair <ll,double>> & VetTruns){
    auto type = GetNetType();
    assert(type == NET_TYPE::GATE);
    for (auto & VetTrun : VetTruns){
        auto consts = CreateConst();
        if (VetTrun.second == -1)
            Replace(VetTrun.first, consts.first);
        else if (VetTrun.second == 1)
            Replace(VetTrun.first, consts.second);
        else
            assert(0);
    }
    CleanUp();
}


void NetMan::MergeConst() {
    pair <ll, ll> ret(-1, -1);
    auto type = GetNetType();
    Abc_Obj_t * pObj = nullptr;
    ll i = 0;
    Abc_NtkForEachNode(GetNet(), pObj, i) {
        if (type == NET_TYPE::GATE || type == NET_TYPE::SOP) {
            if (Abc_NodeIsConst0(pObj)) {
                if (ret.first == -1)
                    ret.first = GetId(pObj);
                else {
                    cout << "merge const 0: " << pObj << endl;
                    Abc_ObjReplace(pObj, GetObj(ret.first));
                }
            }
            else if (Abc_NodeIsConst1(pObj)) {
                if (ret.second == -1)
                    ret.second = GetId(pObj);
                else {
                    cout << "merge const 1: " << pObj << endl;
                    Abc_ObjReplace(pObj, GetObj(ret.second));
                }
            }
        }
        else
            assert(0);
    }
}


void NetMan::ReArrInTopoOrd() {
    AbcMan::SetMainNetw(pNtk); // abc manage the memory of the old network
    AbcMan::TopoSort();
    pNtk = Abc_NtkDup(AbcMan::GetNet()); // NetMan manage the memory of the duplicated network
}


vector <Abc_Obj_t * > NetMan::TopoSort() const {
    vector <Abc_Obj_t *> nodes;
    nodes.reserve(GetNodeNum());
    SetNetNotTrav();
    for (ll i = 0; i < GetPoNum(); ++i) {
        auto pDriver = GetFanin(GetPo(i), 0);
        if (!GetObjTrav(pDriver))
            TopoSortRec(pDriver, nodes);
    }
    return nodes;
}


void NetMan::TopoSortRec(Abc_Obj_t * pObj, vector <Abc_Obj_t *> & nodes) const {
    if (!IsNode(pObj))
        return;
    if (IsConst(pObj))
        return;
    SetObjTrav(pObj);
    for (ll i = 0; i < GetFaninNum(pObj); ++i) {
        auto pFanin = GetFanin(pObj, i);
        if (!GetObjTrav(pFanin))
            TopoSortRec(pFanin, nodes);
    }
    nodes.emplace_back(pObj);
}


vector <ll> NetMan::TopoSortWithIds() const {
    vector <ll> nodes;
    nodes.reserve(GetNodeNum());
    SetNetNotTrav();
    for (ll i = 0; i < GetPoNum(); ++i) {
        auto pDriver = GetFanin(GetPo(i), 0);
        if (!GetObjTrav(pDriver))
            TopoSortRecWithIds(pDriver, nodes);
    }
    return nodes;
}


void NetMan::TopoSortRecWithIds(Abc_Obj_t * pObj, vector <ll> & nodes) const {
    if (!IsNode(pObj))
        return;
    if (IsConst(pObj))
        return;
    SetObjTrav(pObj);
    for (ll i = 0; i < GetFaninNum(pObj); ++i) {
        auto pFanin = GetFanin(pObj, i);
        if (!GetObjTrav(pFanin))
            TopoSortRecWithIds(pFanin, nodes);
    }
    nodes.emplace_back(GetId(pObj));
}


vector <Abc_Obj_t *> NetMan::GetTFI(abc::Abc_Obj_t * pObj) const {
    vector <Abc_Obj_t *> nodes;
    nodes.reserve(GetNodeNum());
    SetNetNotTrav();
    for (ll i = 0; i < GetFaninNum(pObj); ++i) {
        auto pFanin = GetFanin(pObj, i);
        if (!GetObjTrav(pFanin))
            GetTFIRec(pFanin, nodes);
    }
    return nodes;
}


void NetMan::GetTFIRec(abc::Abc_Obj_t * pObj, std::vector <abc::Abc_Obj_t *> & nodes) const {
    if (!IsNode(pObj))
        return;
    SetObjTrav(pObj);
    for (ll i = 0; i < GetFaninNum(pObj); ++i) {
        auto pFanin = GetFanin(pObj, i);
        if (!GetObjTrav(pFanin))
            GetTFIRec(pFanin, nodes);
    }
    nodes.emplace_back(pObj);
}


std::set<ll> NetMan::GetPartialTFI(abc::Abc_Obj_t * pObj, ll TFI_Lev) const {
    std::set<ll> nodes;
    SetNetNotTrav();
    for (ll i = 0; i < GetFaninNum(pObj); ++i) {
        auto pFanin = GetFanin(pObj, i);
        if (!GetObjTrav(pFanin)) {
            GetPartialTFIRec(pFanin, nodes, 1, TFI_Lev);
        }
    }
    return nodes;
}

void NetMan::GetPartialTFIRec(abc::Abc_Obj_t * pObj, std::set<ll> & nodes, ll curLevel, ll maxLevel) const {
    if (!IsNode(pObj))
        return;

    if (curLevel > maxLevel)
        return;

    if (pObj->fMarkA == 1)
        return;

    SetObjTrav(pObj);

    for (ll i = 0; i < GetFaninNum(pObj); ++i) {
        auto pFanin = GetFanin(pObj, i);
        if (!GetObjTrav(pFanin)) {
            GetPartialTFIRec(pFanin, nodes, curLevel + 1, maxLevel);
        }
    }

    nodes.insert(pObj->Id);
}


vector <ll> NetMan::GetTFI(abc::Abc_Obj_t * pObj, const set <ll> & critGraph) const {
    vector <ll> objs;
    objs.reserve(GetNodeNum());
    SetNetNotTrav();
    for (ll i = 0; i < GetFaninNum(pObj); ++i) {
        auto pFanin = GetFanin(pObj, i);
        if (!GetObjTrav(pFanin))
            GetTFIRec(pFanin, objs, critGraph);
    }
    return objs;
}


void NetMan::GetTFIRec(abc::Abc_Obj_t * pObj, std::vector <ll> & objs, const set <ll> & critGraph) const {
    if (critGraph.count(pObj->Id) == 0)
        return;
    // if (!IsNode(pObj))
    //     return;
    SetObjTrav(pObj);
    for (ll i = 0; i < GetFaninNum(pObj); ++i) {
        auto pFanin = GetFanin(pObj, i);
        if (!GetObjTrav(pFanin))
            GetTFIRec(pFanin, objs, critGraph);
    }
    objs.emplace_back(pObj->Id);
}


vector <Abc_Obj_t *> NetMan::GetTFO(abc::Abc_Obj_t * pObj) const {
    vector <Abc_Obj_t *> nodes;
    nodes.reserve(GetNodeNum());
    SetNetNotTrav();
    for (ll i = 0; i < GetFanoutNum(pObj); ++i) {
        auto pFanout = GetFanout(pObj, i);
        if (!GetObjTrav(pFanout))
            GetTFORec(pFanout, nodes);
    }
    reverse(nodes.begin(), nodes.end());
    return nodes;
}


void NetMan::GetTFORec(abc::Abc_Obj_t * pObj, std::vector <abc::Abc_Obj_t *> & nodes) const {
    if (!IsNode(pObj))
        return;
    SetObjTrav(pObj);
    for (ll i = 0; i < GetFanoutNum(pObj); ++i) {
        auto pFanout = GetFanout(pObj, i);
        if (!GetObjTrav(pFanout))
            GetTFORec(pFanout, nodes);
    }
    nodes.emplace_back(pObj);
}

void NetMan::CalcTFO(abc::Abc_Obj_t * pObj, ll & coneSize, ll & nAddNum, bool fModifyMark) const {
    coneSize = 0;
    nAddNum = 0;
    SetNetNotTrav();

    SetObjTrav(pObj);
    ++coneSize;
    if (pObj->fMarkA == 0) {
        ++nAddNum;
        if (fModifyMark)
            pObj->fMarkA = 1;
    }

    for (ll i = 0; i < GetFanoutNum(pObj); ++i) {
        auto pFanout = GetFanout(pObj, i);
        if (!GetObjTrav(pFanout))
            CalcTFORec(pFanout, coneSize, nAddNum, fModifyMark);
    }
}

void NetMan::CalcTFORec(abc::Abc_Obj_t * pObj, ll & coneSize, ll & nAddNum, bool fModifyMark) const {
    if (!IsNode(pObj))
        return;
    SetObjTrav(pObj);
    ++coneSize;
    if (pObj->fMarkA == 0) {
        ++nAddNum;
        if (fModifyMark)
            pObj->fMarkA = 1;
    }
    for (ll i = 0; i < GetFanoutNum(pObj); ++i) {
        auto pFanout = GetFanout(pObj, i);
        if (!GetObjTrav(pFanout))
            CalcTFORec(pFanout, coneSize, nAddNum, fModifyMark);
    }
}

vector <ll> NetMan::GetTFO(abc::Abc_Obj_t * pObj, const set <ll> & critGraph) const {
    vector <ll> objs;
    objs.reserve(GetNodeNum());
    SetNetNotTrav();
    for (ll i = 0; i < GetFanoutNum(pObj); ++i) {
        auto pFanout = GetFanout(pObj, i);
        if (!GetObjTrav(pFanout))
            GetTFORec(pFanout, objs, critGraph);
    }
    reverse(objs.begin(), objs.end());
    return objs;
}


void NetMan::GetTFORec(abc::Abc_Obj_t * pObj, std::vector <ll> & objs, const set <ll> & critGraph) const {
    if (critGraph.count(pObj->Id) == 0)
        return;
    // if (!IsNode(pObj))
    //     return;
    SetObjTrav(pObj);
    for (ll i = 0; i < GetFanoutNum(pObj); ++i) {
        auto pFanout = GetFanout(pObj, i);
        if (!GetObjTrav(pFanout))
            GetTFORec(pFanout, objs, critGraph);
    }
    objs.emplace_back(pObj->Id);
}


void NetMan::Sweep() {
    #ifdef DEBUG
    assert(isDupl == true);
    #endif
    AbcMan::SetMainNetw(pNtk); // abc manage the memory of the old network
    AbcMan::Sweep();
    pNtk = Abc_NtkDup(AbcMan::GetNet()); // NetMan manage the memory of the duplicated network
}


void NetMan::SynthWithResyn2Comm() {
    #ifdef DEBUG
    assert(isDupl == true);
    #endif
    AbcMan::SetMainNetw(pNtk); // abc manage the memory of the old network
    AbcMan::SynthWithResyn2Comm();
    pNtk = Abc_NtkDup(AbcMan::GetNet()); // NetMan manage the memory of the duplicated network
}


void NetMan::SynthAIG() {
    assert(isDupl == true);
    AbcMan::SetMainNetw(pNtk); // abc manage the memory of the old network
    AbcMan::SynthAIG();
    pNtk = Abc_NtkDup(AbcMan::GetNet()); // NetMan manage the memory of the duplicated network
}

void NetMan::ConvToSop() {
    assert(isDupl == true);
    AbcMan::SetMainNetw(pNtk); // abc manage the memory of the old network
    AbcMan::ConvToSop();
    pNtk = Abc_NtkDup(AbcMan::GetNet()); // NetMan manage the memory of the duplicated network
}


void NetMan::ConvToSopWithTwoInps() {
    #ifdef DEBUG
    assert(isDupl == true);
    #endif
    AbcMan::SetMainNetw(pNtk); // abc manage the memory of the old network
    AbcMan::Comm("st; logic; sop;");
    pNtk = Abc_NtkDup(AbcMan::GetNet()); // NetMan manage the memory of the duplicated network
}


// void NetMan::SynthAndMapForSCL(bool isVerb) {
//     #ifdef DEBUG
//     assert(isDupl == true);
//     #endif
//     AbcMan::SetMainNetw(pNtk); // abc manage the memory of the old network
//     AbcMan::SynthAndMap3(isVerb);
//     pNtk = Abc_NtkDup(AbcMan::GetNet()); // NetMan manage the memory of the duplicated network
// }


// void NetMan::SynthAndMapForLUT() {
//     #ifdef DEBUG
//     assert(isDupl == true);
//     #endif
//     AbcMan::SetMainNetw(pNtk); // abc manage the memory of the old network
//     AbcMan::Comm("st; resyn2; dch; if -K 6; sop;");
//     pNtk = Abc_NtkDup(AbcMan::GetNet()); // NetMan manage the memory of the duplicated network
// }


bool NetMan::SynthAndMap(double maxDelay, bool isVerb) {
    #ifdef DEBUG
    assert(isDupl == true);
    #endif
    AbcMan::SetMainNetw(pNtk); // abc manage the memory of the old network
    bool fGenNewSubNodes = AbcMan::SynthAndMap(maxDelay, isVerb);
    pNtk = Abc_NtkDup(AbcMan::GetNet()); // NetMan manage the memory of the duplicated network
    return fGenNewSubNodes;
}

bool NetMan::SynthAndMap_v2(double maxDelay, bool isVerb) {
    #ifdef DEBUG
    assert(isDupl == true);
    #endif
    Abc_Ntk_t * pTmpNtk = Abc_NtkDup(pNtk);
    AbcMan::SetMainNetw(pTmpNtk); 
    AbcMan::SynthAndMap_v2(maxDelay, isVerb, 1);
    // pTmpNtk = Abc_NtkDup(AbcMan::GetNet()); 

    pTmpNtk = Abc_NtkDup(pNtk);
    AbcMan::SetMainNetw(pTmpNtk); 
    AbcMan::SynthAndMap_v2(maxDelay, isVerb, 2);

    pTmpNtk = Abc_NtkDup(pNtk);
    AbcMan::SetMainNetw(pTmpNtk); 
    AbcMan::SynthAndMap_v2(maxDelay, isVerb, 3);

    pTmpNtk = Abc_NtkDup(pNtk);
    AbcMan::SetMainNetw(pTmpNtk); 
    AbcMan::SynthAndMap_v2(maxDelay, isVerb, 4);

    pTmpNtk = Abc_NtkDup(pNtk);
    AbcMan::SetMainNetw(pTmpNtk); 
    AbcMan::SynthAndMap_v2(maxDelay, isVerb, 5);

    pTmpNtk = Abc_NtkDup(pNtk);
    AbcMan::SetMainNetw(pTmpNtk); 
    AbcMan::SynthAndMap_v2(maxDelay, isVerb, 6);

    pTmpNtk = Abc_NtkDup(pNtk);
    AbcMan::SetMainNetw(pTmpNtk); 
    AbcMan::SynthAndMap_v2(maxDelay, isVerb, 7);

    pTmpNtk = Abc_NtkDup(pNtk);
    AbcMan::SetMainNetw(pTmpNtk); 
    AbcMan::SynthAndMap_v2(maxDelay, isVerb, 8);

    pTmpNtk = Abc_NtkDup(pNtk);
    AbcMan::SetMainNetw(pTmpNtk); 
    AbcMan::SynthAndMap_v2(maxDelay, isVerb, 9);

    pTmpNtk = Abc_NtkDup(pNtk);
    AbcMan::SetMainNetw(pTmpNtk); 
    AbcMan::SynthAndMap_v2(maxDelay, isVerb, 10);

    pTmpNtk = Abc_NtkDup(pNtk);
    AbcMan::SetMainNetw(pTmpNtk); 
    AbcMan::SynthAndMap_v2(maxDelay, isVerb, 11);

    pTmpNtk = Abc_NtkDup(pNtk);
    AbcMan::SetMainNetw(pTmpNtk); 
    AbcMan::SynthAndMap_v2(maxDelay, isVerb, 12);

    pTmpNtk = Abc_NtkDup(pNtk);
    AbcMan::SetMainNetw(pTmpNtk); 
    AbcMan::SynthAndMap_v2(maxDelay, isVerb, 13);

    pTmpNtk = Abc_NtkDup(pNtk);
    AbcMan::SetMainNetw(pTmpNtk); 
    AbcMan::SynthAndMap_v2(maxDelay, isVerb, 14);

    return 0;
}


void NetMan::Print(bool showFunct) const {
    if (GetNet()->pName != nullptr)
        cout << GetNet()->pName << endl;
    else
        cout << "network" << endl;
    for (ll i = 0; i < GetIdMaxPlus1(); ++i) {
        if (GetObj(i) == nullptr)
            continue;
        PrintObj(i, showFunct);
    }
}

void NetMan::PrintPro(bool showFunct, bool showFanouts, bool showNode2Po) const {
    if (GetNet()->pName != nullptr)
        cout << GetNet()->pName;
    else
        cout << "network";
    cout << ": total level = " << GetLev() << endl;

    for (ll i = 0; i < GetPoNum(); i++) {
        auto pPo = GetPo(i);
        auto pDriv = Abc_ObjFanin0(pPo);
        pPo->Level = pDriv->Level + 1;
    }

    vector < vector <int>> node2Po;
    if (showNode2Po) {
        node2Po.resize(GetIdMaxPlus1());
        for (ll i = 0; i < GetIdMaxPlus1(); i++) 
            node2Po[i].resize(GetPoNum());
        Abc_Obj_t * pNode;
        ll i;
        for (ll i = 0; i < GetPoNum(); i++) {
            Abc_Obj_t * pPo = GetPo(i);
            node2Po[pPo->Id][i] = 1;
        }
        Abc_NtkForEachNodeReverse(GetNet(), pNode, i) {
            for (ll j = 0; j < GetFanoutNum(i); j++) {
                // ll fanoutId = GetFanoutId(pNode, j);
                ll fanoutId = GetId(GetFanout(i, j));
                if (j == 0) {
                    node2Po[i] = node2Po[fanoutId];
                }
                else {
                    for (ll k = 0; k < GetPoNum(); k++) {
                        node2Po[i][k] |= node2Po[fanoutId][k]; 
                    }
                }
            }
        }
    }

    for (ll i = 0; i < GetIdMaxPlus1(); ++i) {
        if (GetObj(i) == nullptr)
            continue;
        PrintObjPro(i, showFunct, showFanouts);

        if (showNode2Po) {
            for (ll j = 0; j < GetPoNum(); j++) {
                cout << node2Po[i][j];
            }
            cout << endl;
        }
    }
}

void NetMan::PrintLocal(bool showFunct, bool showFanouts, ll startId, ll endId) const {
    if (GetNet()->pName != nullptr)
        cout << GetNet()->pName;
    else
        cout << "network";
    cout << " from Id " << startId << " to " << endId << ":" << endl;

    vector < vector <ll>> PiMarks;
    PiMarks.resize(GetIdMaxPlus1());
    for (ll i = 0; i < GetIdMaxPlus1(); i++) 
        PiMarks[i].resize(GetPiNum());
    Abc_Obj_t * pNode;
    ll i;
    for (ll i = 0; i < GetPiNum(); i++) {
        Abc_Obj_t * pPi = GetPi(i);
        PiMarks[pPi->Id][i] = 1;
    }
    Abc_NtkForEachNode(GetNet(), pNode, i) {
        for (ll j = 0; j < GetFaninNum(i); j++) {
            ll faninId = GetId(GetFanin(i, j));
            if (j == 0) {
                PiMarks[i] = PiMarks[faninId];
            }
            else {
                for (ll k = 0; k < GetPiNum(); k++) {
                    PiMarks[i][k] |= PiMarks[faninId][k]; 
                }
            }
        }
    }

    // if (startId > endId) {
    //     endId = startId;
    //     startId = 0;
    // }
    // for (ll i = startId; i <= endId; ++i) {
    //     if (GetObj(i) == nullptr)
    //         continue;
    //     PrintObjPro(i, showFunct, showFanouts);
    //     for (ll j = 0; j < GetPiNum(); j++)
    //         cout << PiMarks[i][j];
    //     cout << endl;
    // }
    // cout << endl;

    cout << "sub node: ";
    PrintObjPro(startId, showFunct, showFanouts);
    for (ll j = 0; j < GetPiNum(); j++)
        cout << PiMarks[startId][j];
    cout << endl;

    cout << "target node: ";
    PrintObjPro(endId, showFunct, showFanouts);
    for (ll j = 0; j < GetPiNum(); j++)
        cout << PiMarks[endId][j];
    cout << endl;

    cout << "relation of support PI sets of sub node & target node: ";
    if (PiMarks[startId] == PiMarks[endId])
        cout << "==" << endl;
    else {
        bool flag1 = false;
        bool flag2 = false;
        for (ll j = 0; j < GetPiNum(); j++) {
            if (PiMarks[startId][j] && !PiMarks[endId][j])
                flag1 = true;
            else if ((!PiMarks[startId][j]) && PiMarks[endId][j])
                flag2 = true;
        }
        if (flag1 && flag2)
            cout << "intersec" << endl;
        else if (flag1 && !flag2)
            cout << "sub > targ" << endl;
        else if (!flag1 && flag2)
            cout << "sub < targ" << endl;
        else
            cout << "both 0" << endl;
    }

    cout << "fanins of target node:" << endl;
    Abc_Obj_t * pTargNode = GetObj(endId);
    ll faninId;
    Abc_ObjForEachFaninId(pTargNode, faninId, i) {
        PrintObjPro(faninId, showFunct, showFanouts);
        for (ll j = 0; j < GetPiNum(); j++)
            cout << PiMarks[faninId][j];
        cout << endl;
    }
    cout << endl;
}

void NetMan::Print_v2(bool showFunct, bool showFanouts, bool showNode2Po, int metrType, ll nFrame, double errUppBound) const {
    if (GetNet()->pName != nullptr)
        cout << GetNet()->pName;
    else
        cout << "network";
    cout << ": total level = " << GetLev() << endl;

    for (ll i = 0; i < GetPoNum(); i++) {
        auto pPo = GetPo(i);
        auto pDriv = Abc_ObjFanin0(pPo);
        pPo->Level = pDriv->Level + 1;
    }

    vector < vector <int>> node2Po;
    if (showNode2Po) {
        node2Po.resize(GetIdMaxPlus1());
        for (ll i = 0; i < GetIdMaxPlus1(); i++) 
            node2Po[i].resize(GetPoNum());
        Abc_Obj_t * pNode;
        ll i;
        for (ll i = 0; i < GetPoNum(); i++) {
            Abc_Obj_t * pPo = GetPo(i);
            node2Po[pPo->Id][i] = 1;
        }
        Abc_NtkForEachNodeReverse(GetNet(), pNode, i) {
            for (ll j = 0; j < GetFanoutNum(i); j++) {
                // ll fanoutId = GetFanoutId(pNode, j);
                ll fanoutId = GetId(GetFanout(i, j));
                if (j == 0) {
                    node2Po[i] = node2Po[fanoutId];
                }
                else {
                    for (ll k = 0; k < GetPoNum(); k++) {
                        node2Po[i][k] |= node2Po[fanoutId][k]; 
                    }
                }
            }
        }
    }

    for (ll i = 0; i < GetIdMaxPlus1(); ++i) {
        if (GetObj(i) == nullptr)
            continue;
        PrintObj_v2(i, showFunct, showFanouts, metrType, nFrame, errUppBound);

        if (showNode2Po) {
            for (ll j = 0; j < GetPoNum(); j++) {
                cout << node2Po[i][j];
            }
            cout << endl;
        }
    }
}


void NetMan::PrintObjBas(Abc_Obj_t * pObj, string && endWith) const {
    // cout << GetName(pObj) << "(" << GetOriId(pObj) << ")" << endWith;
    cout << GetName(pObj) << "(id = " << pObj->Id << ")" << endWith;
}


void NetMan::PrintObj(Abc_Obj_t * pObj, bool showFunct) const {
    PrintObjBas(pObj, ":");
    for (ll i = 0; i < GetFaninNum(pObj); ++i)
        PrintObjBas(GetFanin(pObj, i), ",");
    // cout << endl;
    if (showFunct) {
        if (NetMan::GetNetType() == NET_TYPE::SOP) {
            if (IsNode(pObj))
                cout << static_cast <char *> (pObj->pData);
            else
                cout << endl;
        }
        else if (NetMan::GetNetType() == NET_TYPE::GATE) {
            if (IsNode(pObj))
                cout << Mio_GateReadName(static_cast <Mio_Gate_t *> (pObj->pData)) << endl;
            else
                cout << endl;
        }
        else
            assert(0);
    }
}

void NetMan::PrintObjPro(Abc_Obj_t * pObj, bool showFunct, bool showFanouts) const {
    PrintObjBas(pObj, " ");
    cout << "lev = " << GetObjLev(pObj) << " ";
    // for (ll i = 0; i < GetFaninNum(pObj); ++i)
    //     PrintObjBas(GetFanin(pObj, i), ",");
    // cout << endl;
    if (showFunct) {
        if (NetMan::GetNetType() == NET_TYPE::SOP) {
            if (IsNode(pObj))
                cout << static_cast <char *> (pObj->pData);
            else
                cout << endl;
        }
        else if (NetMan::GetNetType() == NET_TYPE::GATE) {
            if (IsNode(pObj))
                cout << Mio_GateReadName(static_cast <Mio_Gate_t *> (pObj->pData)) << ") ";
            else
                cout << endl;
        }
        else
            assert(0);
    }

    cout << "fanin(" << GetFaninNum(pObj) << "): ";
        for (ll i = 0; i < GetFaninNum(pObj); ++i)
            cout << GetName(GetFanin(pObj, i)) << " ";

    if (showFanouts) {
        cout << "fanout(" << GetFanoutNum(pObj) << "): ";
        for (ll i = 0; i < GetFanoutNum(pObj); ++i)
            // PrintObjBas(GetFanout(pObj, i), " ");
            cout << GetName(GetFanout(pObj, i)) << " ";
    }
    // cout << "LAC0 error: " << pObj->error0f << ", LAC1 error: " << pObj->error1f;
    // cout << "LAC0 error: " << double(pObj->error0)/double(100032) << ", LAC1 error: " << double(pObj->error1)/double(100032) << " score = " << pObj->markValue;

    cout << endl;
}

void NetMan::PrintObj_v2(Abc_Obj_t * pObj, bool showFunct, bool showFanouts, int metrType, ll nFrame, double errUppBound) const {
    PrintObjBas(pObj, " ");
    // cout << "lev = " << GetObjLev(pObj) << " ";
    // for (ll i = 0; i < GetFaninNum(pObj); ++i)
    //     PrintObjBas(GetFanin(pObj, i), ",");
    // cout << endl;
    if (showFunct) {
        if (NetMan::GetNetType() == NET_TYPE::SOP) {
            if (IsNode(pObj))
                cout << static_cast <char *> (pObj->pData);
            else
                cout << endl;
        }
        else if (NetMan::GetNetType() == NET_TYPE::GATE) {
            if (IsNode(pObj))
                cout << Mio_GateReadName(static_cast <Mio_Gate_t *> (pObj->pData)) << ") ";
            else
                cout << endl;
        }
        else
            assert(0);
    }

    cout << "fanin(" << GetFaninNum(pObj) << "): ";
        for (ll i = 0; i < GetFaninNum(pObj); ++i)
            cout << GetName(GetFanin(pObj, i)) << " ";

    if (showFanouts) {
        cout << "fanout(" << GetFanoutNum(pObj) << "): ";
        for (ll i = 0; i < GetFanoutNum(pObj); ++i)
            // PrintObjBas(GetFanout(pObj, i), " ");
            cout << GetName(GetFanout(pObj, i)) << " ";
    }

    if (metrType == 1) {  // ER
        cout << "LAC0 error: " << double(pObj->error0)/double(nFrame) << ", LAC1 error: " << double(pObj->error1)/double(nFrame) << " score = " << pObj->markValue;
    }
    else if (metrType == 2) {   // MSE
        double err = min(pObj->error0f, pObj->error1f);
        double errNor = log2(sqrt(err) + 1);
        cout << endl << "error = " << err << ", correct score = " << errNor/(errNor+log2(sqrt(errUppBound) + 1)) << ", curr score = " << pObj->markValue;
    }
    else if (metrType == 3) {   // MED
        double err = min(pObj->error0f, pObj->error1f);
        double errNor, errBoundNor;
        if (errUppBound < 300) {
            errNor = log2(err * 1000 + 1);
            errBoundNor = log2(errUppBound * 1000 + 1);
        }
        else {
            errNor = log2(err + 1);
            errBoundNor = log2(errUppBound + 1);
        }
        cout << endl << "error = " << err << ", correct score = " << errNor/(errNor + errBoundNor) << ", curr score = " << pObj->markValue;
    }
    else
        assert(0);

    cout << endl;
}


bool NetMan::IsPIOSame(NetMan & oth_net) const {
    if (this->GetPiNum() != oth_net.GetPiNum())
        return false;
    for (ll i = 0; i < this->GetPiNum(); ++i) {
        if (this->GetPiName(i) != oth_net.GetPiName(i))
            return false;
    }
    if (this->GetPoNum() != oth_net.GetPoNum())
        return false;
    for (ll i = 0; i < this->GetPoNum(); ++i) {
        if (this->GetPoName(i) != oth_net.GetPoName(i))
            return false;
    }
    return true;
}



ll NetMan::GetNodeMffcSize(Abc_Obj_t * pNode) const {
    #ifdef DEBUG
    assert(IsNode(pNode));
    #endif
    Vec_Ptr_t * vCone = Vec_PtrAlloc( 100 );
    Abc_NodeDeref_rec(pNode);
    Abc_NodeMffcConeSupp(pNode, vCone, nullptr);
    Abc_NodeRef_rec( pNode );
    ll ret = Vec_PtrSize(vCone);
    Vec_PtrFree(vCone);
    return ret;
}


vector <Abc_Obj_t *> NetMan::GetNodeMffc(Abc_Obj_t * pNode) const {
    #ifdef DEBUG
    assert(IsNode(pNode));
    #endif
    Vec_Ptr_t * vCone = Vec_PtrAlloc( 100 );
    Abc_NodeDeref_rec(pNode);
    Abc_NodeMffcConeSupp(pNode, vCone, nullptr);
    Abc_NodeRef_rec( pNode );
    vector <Abc_Obj_t *> mffc;
    mffc.reserve(Vec_PtrSize(vCone));
    Abc_Obj_t * pObj = nullptr;
    ll i = 0;
    Vec_PtrForEachEntry(Abc_Obj_t *, vCone, pObj, i)
        mffc.emplace_back(pObj);
    Vec_PtrFree(vCone);
    return mffc;
}

ll NetMan::GetMaxLev(){
    GetLev();
    ll maxlev = 0;
    Abc_Obj_t * pNode = nullptr;
    ll i = 0;
    Abc_NtkForEachNode(GetNet(), pNode, i){
        if (GetObjLev(i) > maxlev)
            maxlev = GetObjLev(i);
    }
    return maxlev;
}

ll NetMan::CreateNode(const std::vector <ll> & faninIds, const std::string & sop) {
    auto pNewNode = abc::Abc_NtkCreateNode(GetNet());
    for (const auto & faninId: faninIds)
        Abc_ObjAddFanin(pNewNode, GetObj(faninId));
    #ifdef DEBUG
    assert(GetNetType() == NET_TYPE::SOP);
    #endif
    pNewNode->pData = Abc_SopRegister((Mem_Flex_t *)GetNet()->pManFunc, sop.c_str());
    return pNewNode->Id;
}


// std::vector <ll> NetMan::TempRepl(abc::Abc_Obj_t * pTS, abc::Abc_Obj_t * pSS) {
//     #ifdef DEBUG
//     assert(pTS != pSS);
//     assert(pTS->pNtk == pSS->pNtk);
//     assert(abc::Abc_ObjFanoutNum(pTS));
//     #endif
//     // record fanouts
//     vector <ll> ret = {pTS->Id, pSS->Id};
//     Abc_Obj_t * pFanout = nullptr;
//     ll i = 0;
//     Abc_ObjForEachFanout(pTS, pFanout, i) {
//         ret.emplace_back(pFanout->Id);
//         ll iFanin = Vec_IntFind(&pFanout->vFanins, pTS->Id);
//         assert(iFanin != -1);
//         ret.emplace_back(iFanin);
//     }
//     PrintVect(ret, "\n");
//     // transfer fanouts
//     abc::Abc_ObjTransferFanout(pTS, pSS);
//     return ret;
// }


static inline int Vec_IntFindFrom(Vec_Int_t * p, int Entry, int start) {
    int i = 0;
    for ( i = start; i < p->nSize; i++ )
        if ( p->pArray[i] == Entry )
            return i;
    assert(0);
    return -1;
}

std::vector <ll> NetMan::TempRepl(abc::Abc_Obj_t * pTS, abc::Abc_Obj_t * pSS) {
    #ifdef DEBUG
    assert(pTS != pSS);
    assert(pTS->pNtk == pSS->pNtk);
    assert(abc::Abc_ObjFanoutNum(pTS));
    #endif
    // record fanouts
    vector <ll> ret = {pTS->Id, pSS->Id};
    Abc_Obj_t * pFanout = nullptr;
    ll i = 0;
    set<pair<int, int>> foIFaninPair;
    Abc_ObjForEachFanout(pTS, pFanout, i) {
        ret.emplace_back(pFanout->Id);
        int start = 0;
        ll iFanin = Vec_IntFindFrom(&pFanout->vFanins, pTS->Id, start);
        while (foIFaninPair.count(pair(pFanout->Id, iFanin)))
            iFanin = Vec_IntFindFrom(&pFanout->vFanins, pTS->Id, iFanin + 1);
        ret.emplace_back(iFanin);
        foIFaninPair.emplace(pair(pFanout->Id, iFanin));
    }
    // PrintVect(ret, "\n");
    // transfer fanouts
    abc::Abc_ObjTransferFanout(pTS, pSS);
    return ret;
}


void NetMan::Recov(std::vector <ll> & replTrace, bool isVerb) {
    #ifdef DEBUG
    assert(replTrace.size() > 2);
    assert(replTrace.size() % 2 == 0);
    #endif
    auto pTS = GetObj(replTrace[0]), pSS = GetObj(replTrace[1]);
    if (isVerb) cout << "recover " << pTS << " and " << pSS << endl;
    for (ll i = 1; i < replTrace.size() / 2; ++i) {
        auto pFanout = GetObj(replTrace[i * 2]);
        auto iFanin = replTrace[i * 2 + 1];
        PatchFanin(pFanout, iFanin, pSS, pTS);
    }
}


static inline int Vec_IntFindRev( Vec_Int_t * p, int Entry ) {
    int i;
    // for ( i = 0; i < p->nSize; i++ )
    for (i = p->nSize - 1; i >= 0; --i)
        if ( p->pArray[i] == Entry )
            return i;
    return -1;
}


static inline int Vec_IntRemoveRev( Vec_Int_t * p, int Entry ) {
    int i;
    // for ( i = 0; i < p->nSize; i++ )
    for (i = p->nSize - 1; i >= 0; --i)
        if ( p->pArray[i] == Entry )
            break;
    if ( i == p->nSize )
        return 0;
    assert( i < p->nSize );
    for ( i++; i < p->nSize; i++ )
        p->pArray[i-1] = p->pArray[i];
    p->nSize--;
    return 1;
}


static inline void Vec_IntPushMem( Mem_Step_t * pMemMan, Vec_Int_t * p, int Entry ) {
    if ( p->nSize == p->nCap )
    {
        int * pArray;
        int i;

        if ( p->nSize == 0 )
            p->nCap = 1;
        if ( pMemMan )
            pArray = (int *)Mem_StepEntryFetch( pMemMan, p->nCap * 8 );
        else
            pArray = ABC_ALLOC( int, p->nCap * 2 );
        if ( p->pArray )
        {
            for ( i = 0; i < p->nSize; i++ )
                pArray[i] = p->pArray[i];
            if ( pMemMan )
                Mem_StepEntryRecycle( pMemMan, (char *)p->pArray, p->nCap * 4 );
            else
                ABC_FREE( p->pArray );
        }
        p->nCap *= 2;
        p->pArray = pArray;
    }
    p->pArray[p->nSize++] = Entry;
}


void NetMan::PatchFanin( Abc_Obj_t * pObj, ll iFanin, Abc_Obj_t * pFaninOld, Abc_Obj_t * pFaninNew ) {
    Abc_Obj_t * pFaninNewR = Abc_ObjRegular(pFaninNew);
    assert( !Abc_ObjIsComplement(pObj) );
    assert( !Abc_ObjIsComplement(pFaninOld) );
    assert( pFaninOld != pFaninNewR );
    assert( pObj->pNtk == pFaninOld->pNtk );
    assert( pObj->pNtk == pFaninNewR->pNtk );
    assert( abc::Abc_ObjFanin(pObj, iFanin) == pFaninOld );

    // remember the attributes of the old fanin
    Vec_IntWriteEntry( &pObj->vFanins, iFanin, pFaninNewR->Id );
    if ( Abc_ObjIsComplement(pFaninNew) )
        Abc_ObjXorFaninC( pObj, iFanin );

    // update the fanout of the fanin
    if ( !Vec_IntRemoveRev( &pFaninOld->vFanouts, pObj->Id ) ) {
        printf( "Node %s is not among", Abc_ObjName(pObj) );
        printf( " the fanouts of its old fanin %s...\n", Abc_ObjName(pFaninOld) );
    }
    Vec_IntPushMem( pObj->pNtk->pMmStep, &pFaninNewR->vFanouts, pObj->Id );
}


void GlobStartAbc() {
    Abc_Start();
    AbcMan abcMan;
    abcMan.LoadAlias();
}


void GlobStopAbc() {
    Abc_Stop();
}


void NetMan::Trunc(ll truncBit) {
    cout << "***** truncate " << truncBit << " bits" << endl;
    // truncation
    auto consts = CreateConst();
    #ifdef DEBUG
    assert(truncBit <= GetPoNum());
    #endif
    for (ll poId = 0; poId < truncBit; ++poId) {
        auto pPo = GetPo(poId);
        #ifdef DEBUG
        assert(GetFaninNum(pPo) == 1);
        #endif
        auto pDriv = GetFanin(pPo, 0);
        if (pDriv != GetObj(consts.first))
            Abc_ObjPatchFanin(pPo, pDriv, GetObj(consts.first));
    }
    // clean up
    CleanUp();
}


void NetMan::SetBit(ll iBit, bool useConst1) {
    // cout << "***** set bit-" << iBit << " to " << useConst1 << endl;
    auto consts = CreateConst();
    assert(iBit <= GetPoNum());
    auto pPo = GetPo(iBit);
    assert(GetFaninNum(pPo) == 1);
    auto pDriv = GetFanin(pPo, 0);
    if (useConst1 && pDriv != GetObj(consts.second))
        Abc_ObjPatchFanin(pPo, pDriv, GetObj(consts.second));
    else if (!useConst1 && pDriv != GetObj(consts.first))
        Abc_ObjPatchFanin(pPo, pDriv, GetObj(consts.first));
    CleanUp();
}

void NetMan::SetBitNode(ll iBit, bool useConst1, ll faninNum) {
    // cout << "***** set bit-" << iBit << " to " << useConst1 << endl;
    auto consts = CreateConst();
    assert(iBit <= GetNodeNum());
    auto pObj = GetObj(iBit);
    auto pDriv = GetFanin(pObj, faninNum);
    Abc_ObjPatchFanin(pObj, pDriv, useConst1? GetObj(consts.second): GetObj(consts.first));
    CleanUp();
}

void NetMan::InvConstNode(bool isConst0, ll fanoutNum) {
    auto consts = CreateConst();
    Abc_Obj_t* pDriv;
    if(isConst0)
        pDriv = GetObj(consts.first);
    else
        pDriv = GetObj(consts.second);
    auto pObj = GetFanout(pDriv, fanoutNum);
    Abc_ObjPatchFanin(pObj, pDriv, isConst0? GetObj(consts.second): GetObj(consts.first));
    CleanUp();
}

void NetMan::InvConstNodeAll(unsigned seed) {
    uniform_int <> unif01(0, 1);
    random::mt19937 eng(seed);
    variate_generator < random::mt19937, uniform_int <> > rand01(eng, unif01);
    auto consts = CreateConst();
    vector <ll> pObj;
    vector <ll> pDriv;
    for (auto i = 0; i < GetFanoutNum(consts.first); ++i){
        auto targId = GetFanoutId(consts.first, i);
        if (IsObjPo(targId))
            continue;
        pDriv.emplace_back(consts.first);
        pObj.emplace_back(targId);
    }
    for (auto j = 0; j < GetFanoutNum(consts.second); ++j){
        auto targId = GetFanoutId(consts.second, j);
        if (IsObjPo(targId))
            continue;
        pDriv.emplace_back(consts.second);
        pObj.emplace_back(targId);
    }
    for (auto k = 0; k < int(pObj.size()); ++k){
        if(rand01())
            Abc_ObjPatchFanin(GetObj(pObj[k]), GetObj(pDriv[k]), IsConst0(pDriv[k])? GetObj(consts.second): GetObj(consts.first));
    }
    CleanUp();
}


bool NetMan::CleanUp() {
    bool isUpd = false;
    bool isCont = true;
    while (isCont) {
        isCont = false;
        // delete redundant node
        for (ll nodeId = 0; nodeId < GetIdMaxPlus1(); ++nodeId) {
            if (IsNode(nodeId) && GetFanoutNum(nodeId) == 0) {
                isCont = true;
                auto mffc = GetNodeMffc(GetObj(nodeId));
                for (const auto & pObj: mffc) {
                    // cout << "delete " << pObj << " ";
                    // if (GetNetType() == NET_TYPE::GATE) {
                    //     auto gateName = GetGateName(pObj);
                    //     cout << gateName;
                    //     if (gateName.find("HA1") != -1 || gateName.find("FA1") != -1) {
                    //         if (GetTwinNode(pObj) != nullptr)
                    //             cout << " twin " << GetTwinNode(pObj);
                    //     }
                    // }
                    // cout << endl;
                    DelObj(pObj);
                    isUpd = true;
                }
                break;
            }
        }
    }
    return isUpd;
}

void NetMan::MergeIdentNode() {
    // merge identical node, to reduce the conflicting when finding twin node
    if (GetNetType() != NET_TYPE::GATE)
        return;
    ll idMaxPlus1 = GetIdMaxPlus1();
    for (ll nodeId = 0; nodeId < idMaxPlus1; ++nodeId) {
        if (!IsNode(nodeId)) continue;
        auto pNode = GetObj(nodeId);
        // if (GetFanoutNum(pNode) != 0){
        if (GetGateName(pNode).find("HA1") != -1) {
            for (auto & ident : GetIdentNode(pNode)){
                // auto p_copy = ident;
                Abc_ObjReplace(ident, pNode);
                // assert(GetFanoutNum(p_copy) == 0);
            }
        }
        else if (GetGateName(pNode).find("FA1") != -1) {
            for (auto & ident : GetIdentNode(pNode)){
                // auto p_copy = ident;
                Abc_ObjReplace(ident, pNode);
                // assert(GetFanoutNum(p_copy) == 0);
            }
        }
        // }
    }
}

void NetMan::BreakModiGateAll(){
    Abc_Obj_t* pObj = nullptr;
    int i = 0;
    Abc_NtkForEachNode(pNtk, pObj, i){
        if (GetGateName(pObj).find("MAOI22D") != -1 || GetGateName(pObj).find("MAOI22D") != -1){
            auto Id = GetId(pObj);
            BreakModiGate(Id);
        }
    }
}

void NetMan::BreakModiGate(ll & targetId){
    assert(IsNode(targetId));
    auto pNode = GetObj(targetId);
    vector <Abc_Obj_t *> fanins;
    ll nFanin = GetFaninNum(pNode);
    for (ll iFanin = 0; iFanin < nFanin; ++iFanin)
        fanins.emplace_back(GetFanin(pNode, iFanin));
    if (GetGateName(pNode).find("MAOI22D") != -1){ //Function = (~(A1A2))(B1+B2)
        assert (nFanin == 4);
        auto pNAND = CreateGate(std::vector <Abc_Obj_t *> ({fanins[0], fanins[1]}), "CKND2D0BWP7T30P140HVT");
        auto pOR = CreateGate(std::vector <Abc_Obj_t *> ({fanins[2], fanins[3]}), "OR2D0BWP7T30P140HVT");
        auto pAND = CreateGate(std::vector <Abc_Obj_t *> ({pNAND, pOR}), "CKAN2D0BWP7T30P140HVT");
        TransfFanout(pNode, pAND);
        CleanUp();
        cout << "detect MAOI22, split it to a NAND gate, an OR gate and an AND gate" << endl;
    }
    else if (GetGateName(pNode).find("MOAI22D") != -1){ //Function = (~(A1+A2))+(B1B2)
        assert (nFanin == 4);
        auto pNOR = CreateGate(std::vector <Abc_Obj_t *> ({fanins[0], fanins[1]}), "NR2D0BWP7T30P140HVT");
        auto pAND = CreateGate(std::vector <Abc_Obj_t *> ({fanins[2], fanins[3]}), "CKAN2D0BWP7T30P140HVT");
        auto pOR = CreateGate(std::vector <Abc_Obj_t *> ({pNOR, pAND}), "OR2D0BWP7T30P140HVT");
        TransfFanout(pNode, pOR);
        CleanUp();
        cout << "detect MOAI22, split it to a NOR gate, an AND gate and an OR gate" << endl;
    }
    else if (GetGateName(pNode).find("MUX2ND") != -1){
        assert (nFanin == 3);
        auto pMUX = CreateGate(std::vector <Abc_Obj_t *> ({fanins[0], fanins[1], fanins[2]}), "MUX2D1BWP7T30P140HVT");
        auto pNOT = CreateGate(std::vector <Abc_Obj_t *> ({pMUX}), "CKND0BWP7T30P140HVT");
        TransfFanout(pNode, pNOT);
        CleanUp();
        cout << "detect MUX2ND, split it to a MUX2 gate, and an Inv" << endl;
    }
    else if (GetGateName(pNode).find("AOI222D") != -1){
        assert (nFanin == 6);
        auto pAO22 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[0], fanins[1], fanins[2], fanins[3]}), "AO22D0BWP7T30P140HVT");
        auto pAOI21 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[4], fanins[5], pAO22}), "AOI21D0BWP7T30P140HVT");
        TransfFanout(pNode, pAOI21);
        CleanUp();
        cout << "detect AOI222, split it to an AO22 gate, and an AOI21 gate" << endl;
    }
    else if (GetGateName(pNode).find("AO222D") != -1){
        assert (nFanin == 6);
        auto pAO22 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[0], fanins[1], fanins[2], fanins[3]}), "AO22D0BWP7T30P140HVT");
        auto pAO21 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[4], fanins[5], pAO22}), "AO21D0BWP7T30P140HVT");
        TransfFanout(pNode, pAO21);
        CleanUp();
        cout << "detect AO222, split it to an AO22 gate, and an AO21 gate" << endl;
    }
    else if (GetGateName(pNode).find("OA222D") != -1){
        assert (nFanin == 6);
        auto pOA22 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[0], fanins[1], fanins[2], fanins[3]}), "OA22D0BWP7T30P140HVT");
        auto pOA21 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[4], fanins[5], pOA22}), "OA21D0BWP7T30P140HVT");
        TransfFanout(pNode, pOA21);
        CleanUp();
        cout << "detect OA222, split it to an OA22 gate, and an OA21 gate" << endl;
    }
    else if (GetGateName(pNode).find("OAI222D") != -1){
        assert (nFanin == 6);
        auto pOA22 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[0], fanins[1], fanins[2], fanins[3]}), "OA22D0BWP7T30P140HVT");
        auto pOAI21 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[4], fanins[5], pOA22}), "OAI21D0BWP7T30P140HVT");
        TransfFanout(pNode, pOAI21);
        CleanUp();
        cout << "detect OAI222, split it to an OA22 gate, and an OAI21 gate" << endl;
    }
    else if (GetGateName(pNode).find("AOI33D") != -1){
        assert (nFanin == 6);
        auto pAND1 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[0], fanins[1], fanins[2]}), "AN3D0BWP7T30P140HVT");
        auto pAND2 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[3], fanins[4], fanins[5]}), "AN3D0BWP7T30P140HVT");
        auto pNOR = CreateGate(std::vector <Abc_Obj_t *> ({pAND1, pAND2}), "NR2D0BWP7T30P140HVT");
        TransfFanout(pNode, pNOR);
        CleanUp();
        cout << "detect AOI33, split it to two AND3 gate, and a NOR gate" << endl;
    }
    else if (GetGateName(pNode).find("AO33D") != -1){
        assert (nFanin == 6);
        auto pAND1 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[0], fanins[1], fanins[2]}), "AN3D0BWP7T30P140HVT");
        auto pAND2 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[3], fanins[4], fanins[5]}), "AN3D0BWP7T30P140HVT");
        auto pOR = CreateGate(std::vector <Abc_Obj_t *> ({pAND1, pAND2}), "OR2D0BWP7T30P140HVT");
        TransfFanout(pNode, pOR);
        CleanUp();
        cout << "detect AO33, split it to two AND3 gate, and an OR gate" << endl;
    }
    else if (GetGateName(pNode).find("OAI33D") != -1){
        assert (nFanin == 6);
        auto pOR1 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[0], fanins[1], fanins[2]}), "OR3D0BWP7T30P140HVT");
        auto pOR2 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[3], fanins[4], fanins[5]}), "OR3D0BWP7T30P140HVT");
        auto pNAND = CreateGate(std::vector <Abc_Obj_t *> ({pOR1, pOR2}), "CKND2D0BWP7T30P140HVT");
        TransfFanout(pNode, pNAND);
        CleanUp();
        cout << "detect OAI33, split it to two OR3 gate, and a NAND gate" << endl;
    }
    else if (GetGateName(pNode).find("OA33D") != -1){
        assert (nFanin == 6);
        auto pOR1 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[0], fanins[1], fanins[2]}), "OR3D0BWP7T30P140HVT");
        auto pOR2 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[3], fanins[4], fanins[5]}), "OR3D0BWP7T30P140HVT");
        auto pAND = CreateGate(std::vector <Abc_Obj_t *> ({pOR1, pOR2}), "CKAN2D0BWP7T30P140HVT");
        TransfFanout(pNode, pAND);
        CleanUp();
        cout << "detect OAI33, split it to two OR3 gate, and an AND gate" << endl;
    }
}

void NetMan::FixConstInp(ll & constId, ll & targetId){
    assert(IsNode(targetId));
    auto pNode = GetObj(targetId);
    bool isC0 = IsConst0(constId);
    auto sop = string(Mio_GateReadSop((Mio_Gate_t *)pNode->pData));
    auto pLib = (Mio_Library_t *)Abc_FrameReadLibGen();
    auto pNewNode = Abc_NtkCreateNode(GetNet()); // pNewNode may not be used.
    Abc_Obj_t * pSub = nullptr;
    Mio_Gate_t * pGate = nullptr;
    auto consts = CreateConst();
    ll SubId = 0;
    // If we need to create new gate, there is(are) fanin(s) to be deleted. 
    if (GetGateName(pNode).find("AOI22D") != -1 || GetGateName(pNode).find("OAI22D") != -1 || GetGateName(pNode).find("OA22D") != -1 || GetGateName(pNode).find("AO22D") != -1){
        bool isA;
        if (isC0)
            isA = IsConst0(GetFanin(pNode,0)) || IsConst0(GetFanin(pNode,1));
        else
            isA = IsConst1(GetFanin(pNode,0)) || IsConst1(GetFanin(pNode,1));
        if (isA){
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
        }
        else{
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
        }
    }
    else if (GetGateName(pNode).find("OAI221D") != -1 || GetGateName(pNode).find("OA221D") != -1){
        bool isA, isB, isC;
        if (isC0){
            isA = IsConst0(GetFanin(pNode,0)) || IsConst0(GetFanin(pNode,1));
            isB = IsConst0(GetFanin(pNode,2)) || IsConst0(GetFanin(pNode,3));
            isC = IsConst0(GetFanin(pNode,4));
        }
        else {
            isA = IsConst1(GetFanin(pNode,0)) || IsConst1(GetFanin(pNode,1));
            isB = IsConst1(GetFanin(pNode,2)) || IsConst1(GetFanin(pNode,3));
            isC = IsConst1(GetFanin(pNode,4));
        }
        if (isA){
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 4));
            if (isC0 && IsConst0(GetFanin(pNode,0)))
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
            if (isC0 && IsConst0(GetFanin(pNode,1)))
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
        }
        if (isB){
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 4));
            if (isC0 && IsConst0(GetFanin(pNode,2)))
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
            if (isC0 && IsConst0(GetFanin(pNode,3)))
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
        }
        if (isC){
            for (auto i = 0; i < 4; ++i)
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, i));
        }
    }
    else if (GetGateName(pNode).find("AOI221D") != -1 || GetGateName(pNode).find("AO221D") != -1){
        bool isA, isB, isC;
        if (isC0){
            isA = IsConst0(GetFanin(pNode,0)) || IsConst0(GetFanin(pNode,1));
            isB = IsConst0(GetFanin(pNode,2)) || IsConst0(GetFanin(pNode,3));
            isC = IsConst0(GetFanin(pNode,4));
        }
        else {
            isA = IsConst1(GetFanin(pNode,0)) || IsConst1(GetFanin(pNode,1));
            isB = IsConst1(GetFanin(pNode,2)) || IsConst1(GetFanin(pNode,3));
            isC = IsConst1(GetFanin(pNode,4));
        }
        if (isA){
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 4));
            if (!isC0 && IsConst1(GetFanin(pNode,0)))
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
            if (!isC0 && IsConst1(GetFanin(pNode,1)))
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
        }
        if (isB){
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 4));
            if (!isC0 && IsConst1(GetFanin(pNode,2)))
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
            if (!isC0 && IsConst1(GetFanin(pNode,3)))
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
        }
        if (isC){
            for (auto i = 0; i < 4; ++i)
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, i));
        }
    }
    else if (GetGateName(pNode).find("OAI211D") != -1 || GetGateName(pNode).find("OA211D") != -1){
        bool isA;
        if (isC0)
            isA = IsConst0(GetFanin(pNode,0)) || IsConst0(GetFanin(pNode,1));
        else
            isA = IsConst1(GetFanin(pNode,0)) || IsConst1(GetFanin(pNode,1));
        if (!isC0 && isA){
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
        }
        else{
            bool delconst = false;
            for (ll faninId = 0; faninId < GetFaninNum(pNode); ++faninId){
                if(isC0){
                    if (!IsConst0(GetFanin(pNode,faninId)) || delconst)
                        Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                    else
                        delconst = true;
                }
                else {
                    if (!IsConst1(GetFanin(pNode,faninId)) || delconst)
                        Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                    else
                        delconst = true;
                }
            }
        }
    }
    else if (GetGateName(pNode).find("AOI211D") != -1 || GetGateName(pNode).find("AO211D") != -1){
        bool isA;
        if (isC0)
            isA = IsConst0(GetFanin(pNode,0)) || IsConst0(GetFanin(pNode,1));
        else
            isA = IsConst1(GetFanin(pNode,0)) || IsConst1(GetFanin(pNode,1));
        if (isC0 && isA){
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
            Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
        }
        else{
            bool delconst = false;
            for (ll faninId = 0; faninId < GetFaninNum(pNode); ++faninId){
                if(isC0){
                    if (!IsConst0(GetFanin(pNode,faninId)) || delconst)
                        Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                    else
                        delconst = true;
                }
                else {
                    if (!IsConst1(GetFanin(pNode,faninId)) || delconst)
                        Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                    else
                        delconst = true;
                }
            }
        }
    }
    else if (GetGateName(pNode).find("OAI32D") != -1 || GetGateName(pNode).find("OA32D") != -1){
        bool isA, isB;
        if (!isC0){
            isA = IsConst1(GetFanin(pNode,0)) || IsConst1(GetFanin(pNode,1)) || IsConst1(GetFanin(pNode,2));
            isB = IsConst1(GetFanin(pNode,3)) || IsConst1(GetFanin(pNode,4));
            if (isA){
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 4));
            }
            else if (isB){
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
            }
        }
        else{
            bool delconst = false;
            for (ll faninId = 0; faninId < GetFaninNum(pNode); ++faninId){
                if(isC0){
                    if (!IsConst0(GetFanin(pNode,faninId)) || delconst)
                        Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                    else
                        delconst = true;
                }
                else {
                    if (!IsConst1(GetFanin(pNode,faninId)) || delconst)
                        Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                    else
                        delconst = true;
                }
            }
        }
    }
    else if (GetGateName(pNode).find("OAI33D") != -1 || GetGateName(pNode).find("OA33D") != -1){
        bool isA;
        if (isC0){
            isA = IsConst0(GetFanin(pNode,0)) || IsConst0(GetFanin(pNode,1)) || IsConst0(GetFanin(pNode,2));
            if (isA) {
                bool isA1 = IsConst0(GetFanin(pNode,0));
                bool isA2 = IsConst0(GetFanin(pNode,1));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 4));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 5));
                if (isA1) {
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
                }
                else if (isA2) {
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
                }
                else {
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
                }
            }
            else {
                bool delconst = false;
                for (ll faninId = 0; faninId < GetFaninNum(pNode); ++faninId){
                    if (!IsConst0(GetFanin(pNode,faninId)) || delconst)
                        Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                    else
                        delconst = true;
                }
            }
        }
        else {
            isA = IsConst1(GetFanin(pNode,0)) || IsConst1(GetFanin(pNode,1)) || IsConst1(GetFanin(pNode,2));
            if (isA) {
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 4));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 5));
            }
            else {
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
            }
        }
    }
    else if (GetGateName(pNode).find("AOI32D") != -1 || GetGateName(pNode).find("AO32D") != -1){
        bool isA, isB;
        if (isC0){
            isA = IsConst0(GetFanin(pNode,0)) || IsConst0(GetFanin(pNode,1)) || IsConst0(GetFanin(pNode,2));
            isB = IsConst0(GetFanin(pNode,3)) || IsConst0(GetFanin(pNode,4));
            if (isA){
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 4));
            }
            else if (isB){
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
            }
        }
        else{
            bool delconst = false;
            for (ll faninId = 0; faninId < GetFaninNum(pNode); ++faninId){
                if(isC0){
                    if (!IsConst0(GetFanin(pNode,faninId)) || delconst)
                        Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                    else
                        delconst = true;
                }
                else {
                    if (!IsConst1(GetFanin(pNode,faninId)) || delconst)
                        Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                    else
                        delconst = true;
                }
            }
        }
    }
    else if (GetGateName(pNode).find("OAI31D") != -1 || GetGateName(pNode).find("OA31D") != -1){
        bool isA, isB;
        if (!isC0){
            isA = IsConst1(GetFanin(pNode,0)) || IsConst1(GetFanin(pNode,1)) || IsConst1(GetFanin(pNode,2));
            isB = IsConst1(GetFanin(pNode,3));
            if (isA){
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
            }
            else if (isB){
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
            }
        }
        else{
            bool delconst = false;
            for (ll faninId = 0; faninId < GetFaninNum(pNode); ++faninId){
                if(isC0){
                    if (!IsConst0(GetFanin(pNode,faninId)) || delconst)
                        Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                    else
                        delconst = true;
                }
                else {
                    if (!IsConst1(GetFanin(pNode,faninId)) || delconst)
                        Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                    else
                        delconst = true;
                }
            }
        }
    }
    else if (GetGateName(pNode).find("AOI31D") != -1 || GetGateName(pNode).find("AO31D") != -1){
        bool isA, isB;
        if (isC0){
            isA = IsConst0(GetFanin(pNode,0)) || IsConst0(GetFanin(pNode,1)) || IsConst0(GetFanin(pNode,2));
            isB = IsConst0(GetFanin(pNode,3));
            if (isA){
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
            }
            else if (isB){
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
            }
        }
        else{
            bool delconst = false;
            for (ll faninId = 0; faninId < GetFaninNum(pNode); ++faninId){
                if(isC0){
                    if (!IsConst0(GetFanin(pNode,faninId)) || delconst)
                        Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                    else
                        delconst = true;
                }
                else {
                    if (!IsConst1(GetFanin(pNode,faninId)) || delconst)
                        Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                    else
                        delconst = true;
                }
            }
        }
    }
    else{
        bool delconst = false;
        for (ll faninId = 0; faninId < GetFaninNum(pNode); ++faninId){
            if(isC0){
                if (!IsConst0(GetFanin(pNode,faninId)) || delconst)
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                else
                    delconst = true;
            }
            else {
                if (!IsConst1(GetFanin(pNode,faninId)) || delconst)
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                else
                    delconst = true;
            }
        }
    }
    if (GetGateName(pNode).find("FA1") != -1) {
        if (sop == "1-1 1\n-11 1\n11- 1\n") { // CO=A B+B CI+A CI
            if (isC0)
                pGate = Mio_LibraryReadGateByName(pLib, "HA1D1BWP7T30P140HVT", "CO"); //Function = A + B + 0 = A + B; CO = A&B use HA to replace, if no twin, update in ProcHAFA
            else
                pGate = Mio_LibraryReadGateByName(pLib, "OR2D0BWP7T30P140HVT", nullptr); //Function = A + B + 1; CO = A|B 
            assert(pGate != nullptr);
            pNewNode->pData = pGate;
            pSub = pNewNode;
            if (isC0)
                cout << "replace (FA-CO) " << pNode << " with new node " << pSub << " and new Fun HA-CO" << endl;
            else
                cout << "replace (FA-CO) " << pNode << " with new node " << pSub << " and new Fun OR2" << endl;
        }
        else if (sop == "100 1\n010 1\n111 1\n001 1\n") { // S=A^B^CI
            if (isC0)
                pGate = Mio_LibraryReadGateByName(pLib, "HA1D1BWP7T30P140HVT", "S"); //Function = A + B + 0 = A + B; S = A^B use HA to replace, if no twin, update in ProcHAFA
            else
                pGate = Mio_LibraryReadGateByName(pLib, "XNR2D0BWP7T30P140HVT", nullptr); //Function = A + B + 1; S = ~(A^B)
            assert(pGate != nullptr);
            pNewNode->pData = pGate;
            pSub = pNewNode;
            if (isC0)
                cout << "replace (FA-S) " << pNode << " with new node " << pSub << " and new Fun HA-S" << endl;
            else
                cout << "replace (FA-S) " << pNode << " with new node " << pSub << " and new Fun XNOR2" << endl;

        }
        else {
            cout << sop;
            assert(0);
        }
        assert(pSub != nullptr);
        TransfFanout(pNode, pSub);
        DelObj(pNode);
    }
    else if (GetGateName(pNode).find("HA1") != -1) {
        if (sop == "11 1\n") { // CO=A&B
            if (isC0){
                cout << "replace (HA-CO) " << pNode << " with const 0 " << endl;
                Replace(targetId, consts.first); //Function = A + 0; CO = 0
            }
            else { // Function = A + 1; CO = A
                if (IsConst1(GetFaninId(targetId, 0))) //check whether the first input is 1
                    SubId = GetFaninId(targetId, 1); 
                else
                    SubId = GetFaninId(targetId, 0); 
                cout << "replace (HA-CO) " << pNode << " with its fanin " << GetObj(SubId) << endl;
                Replace(targetId, SubId);
            }
        }
        else if (sop == "10 1\n01 1\n" || sop == "01 1\n10 1\n") { // S=A^B
            if (isC0){ // Function = A + 0; S = A;
                if (IsConst0(GetFaninId(targetId, 0))) //check whether the first input is 0
                    SubId = GetFaninId(targetId, 1); 
                else
                    SubId = GetFaninId(targetId, 0); 
                cout << "replace (HA-S) " << pNode << " with its fanin " << GetObj(SubId) << endl;
                Replace(targetId, SubId); 
            }
            else { // Function = A + 1; S = ~A;
                if (IsConst1(GetFaninId(targetId, 0)))
                    pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 1));
                else
                    pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 0));
                SubId = GetId(pSub);
                cout << "replace (HA-S) " << pNode << " with its fanin (an Inv of old fanin) " << pSub << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
    }
    else if (GetGateName(pNode).find("MUX2D") != -1){   // Function = MUX2
        bool isA0 = IsConst0(GetFanin(pNode,0));
        bool isA1 = IsConst1(GetFanin(pNode,0));
        bool isB0 = IsConst0(GetFanin(pNode,1));
        bool isS0 = IsConst0(GetFanin(pNode,2));
        bool isS1 = IsConst1(GetFanin(pNode,2));
        if (isS0 && isC0){
            SubId = GetFaninId(targetId, 0);  // Function = A
            cout << "replace (MUX2) " << pNode << " with its first fanin " << GetObj(SubId) << endl;
            Replace(targetId, SubId);
        }
        else if (isS1 && !isC0){
            SubId = GetFaninId(targetId, 1);  // Function = B
            cout << "replace (MUX2) " << pNode << " with its second fanin " << GetObj(SubId) << endl;
            Replace(targetId, SubId);
        }
        else if (isA0 && isC0){
            pGate = Mio_LibraryReadGateByName(pLib, "CKAN2D0BWP7T30P140HVT", nullptr); // Only B = S = 1 Function = 1
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (MUX2) " << pNode << " with new node " << pSub << " and new Fun AND2" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
        else if (isA1 && !isC0){
            pGate = Mio_LibraryReadGateByName(pLib, "IND2D0BWP7T30P140HVT", nullptr); // Only B = 0 S = 1 Function = 0
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (MUX2) " << pNode << " with new node " << pSub << " and new Fun INAND2" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
        else if (isB0 && isC0){
            pGate = Mio_LibraryReadGateByName(pLib, "INR2D0BWP7T30P140HVT", nullptr); // Only A = 1 S = 0 Function = 1
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (MUX2) " << pNode << " with new node " << pSub << " and new Fun INOR2" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
        else{
            pGate = Mio_LibraryReadGateByName(pLib, "OR2D0BWP7T30P140HVT", nullptr); // Only A = 0 S = 0 Function = 0
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (MUX2) " << pNode << " with new node " << pSub << " and new Fun OR2" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
    }
    else if (GetGateName(pNode).find("MAOI222D") != -1){// Function = ~(AB + BC + CA)
        if (isC0)
            pGate = Mio_LibraryReadGateByName(pLib, "CKND2D0BWP7T30P140HVT", nullptr); // Function = ~(AB)
        else
            pGate = Mio_LibraryReadGateByName(pLib, "NR2D0BWP7T30P140HVT", nullptr); // Function = ~(AB + A + B) = ~(A + B)
        pNewNode->pData = pGate;
        pSub = pNewNode;
        std::string Fun = isC0? "NAND2" : "NOR2";
        cout << "replace (MAOI222) " << pNode << " with new node " << pSub << " and new Fun " << Fun << endl;
        TransfFanout(pNode, pSub);
        DelObj(pNode);
    }
    else if (GetGateName(pNode).find("MUX2ND") != -1 || GetGateName(pNode).find("OAI222D") != -1 || GetGateName(pNode).find("OA222D") != -1 || GetGateName(pNode).find("AOI222D") != -1 || GetGateName(pNode).find("AO222D") != -1 || GetGateName(pNode).find("OAI33D") != -1 || GetGateName(pNode).find("OA33D") != -1 || GetGateName(pNode).find("AOI33D") != -1 || GetGateName(pNode).find("AO33D") != -1){
        BreakModiGate(targetId);
    }
    else if (GetGateName(pNode).find("AOI32D") != -1){
        if (isC0){
            bool isB = IsConst0(GetFanin(pNode,3)) || IsConst0(GetFanin(pNode,4));
            if (isB){
                pGate = Mio_LibraryReadGateByName(pLib, "ND3D0BWP7T30P140HVT", nullptr); 
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI32) " << pNode << " with new node " << pSub << " and new Fun NAND3" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else {
                pGate = Mio_LibraryReadGateByName(pLib, "ND2D0BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI32) " << pNode << " with new node " << pSub << " and new Fun NAND2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else {
            bool isB = IsConst1(GetFanin(pNode,3)) || IsConst1(GetFanin(pNode,4));
            if (isB){
                pGate = Mio_LibraryReadGateByName(pLib, "AOI31D1BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI32) " << pNode << " with new node " << pSub << " and new Fun AOI31" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else {
                pGate = Mio_LibraryReadGateByName(pLib, "AOI22D1BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI32) " << pNode << " with new node " << pSub << " and new Fun AOI22" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
    }
    else if (GetGateName(pNode).find("AO32D") != -1){
        if (isC0){
            bool isB = IsConst0(GetFanin(pNode,3)) || IsConst0(GetFanin(pNode,4));
            if (isB){
                pGate = Mio_LibraryReadGateByName(pLib, "AN3D0BWP7T30P140HVT", nullptr); 
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AO32) " << pNode << " with new node " << pSub << " and new Fun AND3" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else {
                pGate = Mio_LibraryReadGateByName(pLib, "CKAN2D0BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AO32) " << pNode << " with new node " << pSub << " and new Fun AND2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else {
            bool isB = IsConst1(GetFanin(pNode,3)) || IsConst1(GetFanin(pNode,4));
            if (isB){
                pGate = Mio_LibraryReadGateByName(pLib, "AO31D1BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AO32) " << pNode << " with new node " << pSub << " and new Fun AO31" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else {
                pGate = Mio_LibraryReadGateByName(pLib, "AO22D1BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AO32) " << pNode << " with new node " << pSub << " and new Fun AOI22" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
    }
    else if (GetGateName(pNode).find("OAI33D") != -1){
        if (isC0) {
            pGate = Mio_LibraryReadGateByName(pLib, "OAI32D0BWP7T30P140HVT", nullptr); 
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (OAI33) " << pNode << " with new node " << pSub << " and new Fun OAI32" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
        else {
            pGate = Mio_LibraryReadGateByName(pLib, "NR3D0BWP7T30P140HVT", nullptr); 
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (OAI33) " << pNode << " with new node " << pSub << " and new Fun NOR3" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
    }
    else if (GetGateName(pNode).find("OA33D") != -1){
        if (isC0) {
            pGate = Mio_LibraryReadGateByName(pLib, "OA32D0BWP7T30P140HVT", nullptr); 
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (OA33) " << pNode << " with new node " << pSub << " and new Fun OA32" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
        else {
            pGate = Mio_LibraryReadGateByName(pLib, "OR3D0BWP7T30P140HVT", nullptr); 
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (OA33) " << pNode << " with new node " << pSub << " and new Fun OR3" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
    }
    else if (GetGateName(pNode).find("OAI32D") != -1){
        if (!isC0){
            bool isB = IsConst1(GetFanin(pNode,3)) || IsConst1(GetFanin(pNode,4));
            if (isB){
                pGate = Mio_LibraryReadGateByName(pLib, "NR3D0BWP7T30P140HVT", nullptr); 
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI32) " << pNode << " with new node " << pSub << " and new Fun NOR3" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else {
                pGate = Mio_LibraryReadGateByName(pLib, "NR2D0BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI32) " << pNode << " with new node " << pSub << " and new Fun NOR2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else {
            bool isB = IsConst0(GetFanin(pNode,3)) || IsConst0(GetFanin(pNode,4));
            if (isB){
                pGate = Mio_LibraryReadGateByName(pLib, "OAI31D1BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI32) " << pNode << " with new node " << pSub << " and new Fun OAI31" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else {
                pGate = Mio_LibraryReadGateByName(pLib, "OAI22D1BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI32) " << pNode << " with new node " << pSub << " and new Fun OAI22" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
    }
    else if (GetGateName(pNode).find("OA32D") != -1){
        if (!isC0){
            bool isB = IsConst1(GetFanin(pNode,3)) || IsConst1(GetFanin(pNode,4));
            if (isB){
                pGate = Mio_LibraryReadGateByName(pLib, "OR3D0BWP7T30P140HVT", nullptr); 
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OA32) " << pNode << " with new node " << pSub << " and new Fun OR3" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else {
                pGate = Mio_LibraryReadGateByName(pLib, "OR2D0BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OA32) " << pNode << " with new node " << pSub << " and new Fun OR2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else {
            bool isB = IsConst0(GetFanin(pNode,3)) || IsConst0(GetFanin(pNode,4));
            if (isB){
                pGate = Mio_LibraryReadGateByName(pLib, "OA31D1BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OA32) " << pNode << " with new node " << pSub << " and new Fun OA31" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else {
                pGate = Mio_LibraryReadGateByName(pLib, "OA22D1BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OA32) " << pNode << " with new node " << pSub << " and new Fun OA22" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
    }
    else if (GetGateName(pNode).find("AOI31D") != -1){
        if (isC0){
            bool isB = IsConst0(GetFanin(pNode,3));
            if (isB){
                pGate = Mio_LibraryReadGateByName(pLib, "ND3D0BWP7T30P140HVT", nullptr); 
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI31) " << pNode << " with new node " << pSub << " and new Fun NAND3" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 3));
                SubId = GetId(pSub);
                cout << "replace (AOI31) " << pNode << " with its fanin (an Inv of old fanin) " << pSub << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else {
            bool isB = IsConst1(GetFanin(pNode,3));
            if (isB){
                cout << "replace (AOI31) " << pNode << " with const 0 " << endl;
                Replace(targetId, consts.first); 
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "AOI21D1BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI31) " << pNode << " with new node " << pSub << " and new Fun AOI21" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
    }
    else if (GetGateName(pNode).find("AO31D") != -1){
        if (isC0){
            bool isB = IsConst0(GetFanin(pNode,3));
            if (isB){
                pGate = Mio_LibraryReadGateByName(pLib, "AN3D0BWP7T30P140HVT", nullptr); 
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI31) " << pNode << " with new node " << pSub << " and new Fun AND3" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                SubId = GetFaninId(targetId, 3);
                cout << "replace (AO31) " << pNode << " with its fanin " << GetObj(SubId) << endl;
                Replace(targetId, SubId);
            }
        }
        else {
            bool isB = IsConst1(GetFanin(pNode,3));
            if (isB){
                cout << "replace (AO31) " << pNode << " with const 1 " << endl;
                Replace(targetId, consts.second); 
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "AO21D1BWP7T30P140HVT", nullptr); 
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AO31) " << pNode << " with new node " << pSub << " and new Fun AO21" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
    }
    else if (GetGateName(pNode).find("OAI31D") != -1){
        if (isC0){
            bool isB = IsConst0(GetFanin(pNode,3));
            if (isB){
                cout << "replace (OAI31) " << pNode << " with const 1 " << endl;
                Replace(targetId, consts.second); 
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "OAI21D1BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI31) " << pNode << " with new node " << pSub << " and new Fun OAI21" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else {
            bool isB = IsConst1(GetFanin(pNode,3));
            if (isB){
                pGate = Mio_LibraryReadGateByName(pLib, "NR3D0BWP7T30P140HVT", nullptr); 
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI31) " << pNode << " with new node " << pSub << " and new Fun NOR3" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 3));
                SubId = GetId(pSub);
                cout << "replace (OAI31) " << pNode << " with its fanin (an Inv of old fanin) " << pSub << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
    }
    else if (GetGateName(pNode).find("OA31D") != -1){
        if (isC0){
            bool isB = IsConst0(GetFanin(pNode,3));
            if (isB){
                cout << "replace (OAI31) " << pNode << " with const 0 " << endl;
                Replace(targetId, consts.first); 
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "OA21D1BWP7T30P140HVT", nullptr);
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OA31) " << pNode << " with new node " << pSub << " and new Fun OA21" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else {
            bool isB = IsConst1(GetFanin(pNode,3));
            if (isB){
                pGate = Mio_LibraryReadGateByName(pLib, "OR3D0BWP7T30P140HVT", nullptr); 
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OA31) " << pNode << " with new node " << pSub << " and new Fun OR3" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                SubId = GetFaninId(targetId, 3);
                cout << "replace (OA31) " << pNode << " with its fanin " << GetObj(SubId) << endl;
                Replace(targetId, SubId);
            }
        }
    }
    else if (GetGateName(pNode).find("AOI221D") != -1){ // Function = ~(A1A2+B1B2+C)
        if (isC0){
            bool isC = IsConst0(GetFanin(pNode,4));
            if (isC){
                pGate = Mio_LibraryReadGateByName(pLib, "AOI22D1BWP7T30P140HVT", nullptr); // Function = ~(A1A2+B1B2)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI221) " << pNode << " with new node " << pSub << " and new Fun AOI22" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "AOI21D1BWP7T30P140HVT", nullptr); // Function = ~(B1B2+C)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI221) " << pNode << " with new node " << pSub << " and new Fun AOI21" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else{
            bool isA = IsConst1(GetFanin(pNode,0)) || IsConst1(GetFanin(pNode,1));
            bool isB = IsConst1(GetFanin(pNode,2)) || IsConst1(GetFanin(pNode,3));
            bool isC = IsConst1(GetFanin(pNode,4));
            if (isA || isB){
                pGate = Mio_LibraryReadGateByName(pLib, "AOI211D1BWP7T30P140HVT", nullptr); // Function = ~(B1B2+C+A1)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI221) " << pNode << " with new node " << pSub << " and new Fun AOI211" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            if (isC){
                cout << "replace (AOI221) " << pNode << " with const 0 " << endl;
                Replace(targetId, consts.first); //Function = ~1 = 0
            }
        }
    }
    else if (GetGateName(pNode).find("AO221D") != -1){  // Function = A1A2+B1B2+C
        if (isC0){
            bool isC = IsConst0(GetFanin(pNode,4));
            if (isC){
                pGate = Mio_LibraryReadGateByName(pLib, "AO22D1BWP7T30P140HVT", nullptr); // Function = A1A2+B1B2
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AO221) " << pNode << " with new node " << pSub << " and new Fun AO22" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "AO21D1BWP7T30P140HVT", nullptr); // Function = B1B2+C
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AO221) " << pNode << " with new node " << pSub << " and new Fun AO21" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else{
            bool isA = IsConst1(GetFanin(pNode,0)) || IsConst1(GetFanin(pNode,1));
            bool isB = IsConst1(GetFanin(pNode,2)) || IsConst1(GetFanin(pNode,3));
            bool isC = IsConst1(GetFanin(pNode,4));
            if (isA || isB){
                pGate = Mio_LibraryReadGateByName(pLib, "AO211D1BWP7T30P140HVT", nullptr); // Function = B1B2+C+A1
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AO221) " << pNode << " with new node " << pSub << " and new Fun AO211" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            if (isC){
                cout << "replace (AO221) " << pNode << " with const 1 " << endl;
                Replace(targetId, consts.second); //Function = 1
            }
        }
    } 
    else if (GetGateName(pNode).find("AOI22D") != -1){  // Function = ~(A1A2 + B1B2)
        if (GetGateName(pNode).find("MAOI22D") != -1){
            BreakModiGate(targetId);
        }
        else{
            if (isC0){
                pGate = Mio_LibraryReadGateByName(pLib, "CKND2D0BWP7T30P140HVT", nullptr); // Function = ~(A1A2)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI22) " << pNode << " with new node " << pSub << " and new Fun NAND2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                bool isA1 = IsConst1(GetFanin(pNode,0));
                bool isA2 = IsConst1(GetFanin(pNode,1));
                bool isB1 = IsConst1(GetFanin(pNode,2));
                bool isB2 = IsConst1(GetFanin(pNode,3));
                if (isA1)
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
                else if (isA2)
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
                else if (isB1)
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
                else if (isB2)
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
                pGate = Mio_LibraryReadGateByName(pLib, "AOI21D1BWP7T30P140HVT", nullptr); // Function = ~(A1A2+B1)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI22) " << pNode << " with new node " << pSub << " and new Fun AOI21" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }  
        }
    }
    else if (GetGateName(pNode).find("AO22D") != -1){   // Function = (A1A2 + B1B2)
        if (isC0){
            pGate = Mio_LibraryReadGateByName(pLib, "CKAN2D0BWP7T30P140HVT", nullptr); // Function = A1A2
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (AO22) " << pNode << " with new node " << pSub << " and new Fun AND2" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
        else{
            bool isA1 = IsConst1(GetFanin(pNode,0));
            bool isA2 = IsConst1(GetFanin(pNode,1));
            bool isB1 = IsConst1(GetFanin(pNode,2));
            bool isB2 = IsConst1(GetFanin(pNode,3));
            if (isA1)
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
            else if (isA2)
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
            else if (isB1)
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
            else if (isB2)
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
            pGate = Mio_LibraryReadGateByName(pLib, "AO21D1BWP7T30P140HVT", nullptr); // Function = ~(A1A2+B1)
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (AO22) " << pNode << " with new node " << pSub << " and new Fun AO21" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
    }
    else if (GetGateName(pNode).find("AOI21D") != -1){  // Function = ~(A1A2 + B)
        if (isC0){
            bool isB = IsConst0(GetFanin(pNode,2));
            if (isB){
                pGate = Mio_LibraryReadGateByName(pLib, "CKND2D0BWP7T30P140HVT", nullptr); // Function = ~(A1A2)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI21) " << pNode << " with new node " << pSub << " and new Fun NAND2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 2)); // Function = ~B
                SubId = GetId(pSub);
                cout << "replace (AOI21) " << pNode << " with its fanin (an Inv of B) " << pSub << endl;
                Replace(targetId, SubId);
            }
        }
        else{
            bool isB = IsConst1(GetFanin(pNode,2));
            if (isB){
                cout << "replace (AOI21) " << pNode << " with const 0 " << endl;
                Replace(targetId, consts.first); //Function = ~1 = 0
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "NR2D0BWP7T30P140HVT", nullptr); // Function = ~(A1+B)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI21) " << pNode << " with new node " << pSub << " and new Fun NOR2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
    }
    else if (GetGateName(pNode).find("AO21D") != -1){   // Function = A1A2 + B
        if (isC0){
            bool isB = IsConst0(GetFanin(pNode,2));
            if (isB){
                pGate = Mio_LibraryReadGateByName(pLib, "CKAN2D0BWP7T30P140HVT", nullptr); // Function = A1A2
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AO21) " << pNode << " with new node " << pSub << " and new Fun AND2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                SubId = GetFaninId(targetId, 0);  // Function = B
                cout << "replace (AO21) " << pNode << " with its fanin (B) " << GetObj(SubId) << endl;
                Replace(targetId, SubId);
            }
        }
        else{
            bool isB = IsConst1(GetFanin(pNode,2));
            if (isB){
                cout << "replace (AO21) " << pNode << " with const 1 " << endl;
                Replace(targetId, consts.second); //Function = 1
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "OR2D0BWP7T30P140HVT", nullptr); // Function = (A1+B)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AO21) " << pNode << " with new node " << pSub << " and new Fun OR2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
    }
    else if (GetGateName(pNode).find("AOI211D") != -1){ // Function = ~(A1A2+B+C)
        if (isC0){
            bool isA1 = IsConst0(GetFanin(pNode,0));
            bool isA2 = IsConst0(GetFanin(pNode,1));
            if(isA1 || isA2){
                pGate = Mio_LibraryReadGateByName(pLib, "NR2D0BWP7T30P140HVT", nullptr); // Function = ~(B+C)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI211) " << pNode << " with new node " << pSub << " and new Fun NOR2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "AOI21D1BWP7T30P140HVT", nullptr); // Function = ~(A1A2+B)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI211) " << pNode << " with new node " << pSub << " and new Fun AOI21" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else{
            bool isA1 = IsConst1(GetFanin(pNode,0));
            bool isA2 = IsConst1(GetFanin(pNode,1));
            if(isA1 || isA2){
                pGate = Mio_LibraryReadGateByName(pLib, "NR3D0BWP7T30P140HVT", nullptr); // Function = ~(A1+B+C)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AOI211) " << pNode << " with new node " << pSub << " and new Fun NOR3" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                cout << "replace (AOI211) " << pNode << " with const 0 " << endl;
                Replace(targetId, consts.first); //Function = 0
            }
        }
    }
    else if (GetGateName(pNode).find("AO211D") != -1){  // Function = A1A2+B+C
        if (isC0){
            bool isA1 = IsConst0(GetFanin(pNode,0));
            bool isA2 = IsConst0(GetFanin(pNode,1));
            if(isA1 || isA2){
                pGate = Mio_LibraryReadGateByName(pLib, "OR2D0BWP7T30P140HVT", nullptr); // Function = B+C
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AO211) " << pNode << " with new node " << pSub << " and new Fun OR2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "AO21D1BWP7T30P140HVT", nullptr); // Function = A1A2+B
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AO211) " << pNode << " with new node " << pSub << " and new Fun AO21" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else{
            bool isA1 = IsConst1(GetFanin(pNode,0));
            bool isA2 = IsConst1(GetFanin(pNode,1));
            if(isA1 || isA2){
                pGate = Mio_LibraryReadGateByName(pLib, "OR3D0BWP7T30P140HVT", nullptr); // Function = A1+B+C
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (AO211) " << pNode << " with new node " << pSub << " and new Fun OR3" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                cout << "replace (AO211) " << pNode << " with const 1 " << endl;
                Replace(targetId, consts.second); //Function = 1
            }
        }
    }
    else if (GetGateName(pNode).find("OAI221D") != -1){ // Function = ~((A1+A2)(B1+B2)C)
        if (isC0){
            bool isC = IsConst0(GetFanin(pNode,4));
            if (isC){
                cout << "replace (OAI221) " << pNode << " with const 1 " << endl;
                Replace(targetId, consts.second); //Function = 1
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "OAI211D1BWP7T30P140HVT", nullptr); // Function = ~((A1+A2)BC)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI221) " << pNode << " with new node " << pSub << " and new Fun OAI211" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else{
            bool isA = IsConst1(GetFanin(pNode,0)) || IsConst1(GetFanin(pNode,1));
            bool isB = IsConst1(GetFanin(pNode,2)) || IsConst1(GetFanin(pNode,3));
            bool isC = IsConst1(GetFanin(pNode,4));
            if (isA || isB){
                pGate = Mio_LibraryReadGateByName(pLib, "OAI21D1BWP7T30P140HVT", nullptr); // Function = ~((A1+A2)C)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI221) " << pNode << " with new node " << pSub << " and new Fun OAI21" << endl;
            }
            if (isC){
                pGate = Mio_LibraryReadGateByName(pLib, "OAI22D1BWP7T30P140HVT", nullptr); // Function = ~((A1+A2)(B1+B2))
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI221) " << pNode << " with new node " << pSub << " and new Fun OAI22" << endl;
            }
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
    }
    else if (GetGateName(pNode).find("OA221D") != -1){  // Function = (A1+A2)(B1+B2)C
        if (isC0){
            bool isC = IsConst0(GetFanin(pNode,4));
            if (isC){
                cout << "replace (OAI221) " << pNode << " with const 0 " << endl;
                Replace(targetId, consts.first); //Function = 0
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "OA211D1BWP7T30P140HVT", nullptr); // Function = (A1+A2)BC
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OA221) " << pNode << " with new node " << pSub << " and new Fun OA211" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else{
            bool isA = IsConst1(GetFanin(pNode,0)) || IsConst1(GetFanin(pNode,1));
            bool isB = IsConst1(GetFanin(pNode,2)) || IsConst1(GetFanin(pNode,3));
            bool isC = IsConst1(GetFanin(pNode,4));
            if (isA || isB){
                pGate = Mio_LibraryReadGateByName(pLib, "OA21D1BWP7T30P140HVT", nullptr); // Function = (A1+A2)C
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OA221) " << pNode << " with new node " << pSub << " and new Fun OA21" << endl;
            }
            if (isC){
                pGate = Mio_LibraryReadGateByName(pLib, "AO22D1BWP7T30P140HVT", nullptr); // Function = (A1+A2)(B1+B2)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OA221) " << pNode << " with new node " << pSub << " and new Fun AO22" << endl;
            }
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
    }
    else if (GetGateName(pNode).find("OAI22D") != -1){  // Function = ~((A1+A2)(B1+B2))
        if (GetGateName(pNode).find("MOAI22D") != -1){
            BreakModiGate(targetId);
        }
        else{
            if (isC0){
                bool isA1 = IsConst0(GetFanin(pNode,0));
                bool isA2 = IsConst0(GetFanin(pNode,1));
                bool isB1 = IsConst0(GetFanin(pNode,2));
                bool isB2 = IsConst0(GetFanin(pNode,3));
                if (isA1)
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
                else if (isA2)
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
                else if (isB1)
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
                else if (isB2)
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
                pGate = Mio_LibraryReadGateByName(pLib, "OAI21D1BWP7T30P140HVT", nullptr); // Function = ~((B1+B2)A1)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI22) " << pNode << " with new node " << pSub << " and new Fun OAI21" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "NR2D0BWP7T30P140HVT", nullptr); // Function = ~(B1+B2)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI22) " << pNode << " with new node " << pSub << " and new Fun NOR2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
    }
    else if (GetGateName(pNode).find("OA22D") != -1){   // Function = ((A1+A2)(B1+B2))
        if (isC0){
            bool isA1 = IsConst0(GetFanin(pNode,0));
            bool isA2 = IsConst0(GetFanin(pNode,1));
            bool isB1 = IsConst0(GetFanin(pNode,2));
            bool isB2 = IsConst0(GetFanin(pNode,3));
            if (isA1)
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1));
            else if (isA2)
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0));
            else if (isB1)
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 3));
            else if (isB2)
                Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 2));
            pGate = Mio_LibraryReadGateByName(pLib, "OA21D1BWP7T30P140HVT", nullptr); // Function = (B1+B2)A1
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (OA22) " << pNode << " with new node " << pSub << " and new Fun OA21" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
        else{
            pGate = Mio_LibraryReadGateByName(pLib, "OR2D0BWP7T30P140HVT", nullptr); // Function = B1+B2
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (OA22) " << pNode << " with new node " << pSub << " and new Fun OR2" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
    }
    else if (GetGateName(pNode).find("OAI21D") != -1){  // Function = ~((A1+A2)B)
        bool isB1 = IsConst1(GetFanin(pNode,2));
        bool isB0 = IsConst0(GetFanin(pNode,2));
        if (isC0){
            if (isB0){
                cout << "replace (OAI21) " << pNode << " with const 1 " << endl;
                Replace(targetId, consts.second); //Function = ~0 = 1
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "CKND2D0BWP7T30P140HVT", nullptr); // Function = ~(A1B)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI21) " << pNode << " with new node " << pSub << " and new Fun NAND2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else{
            if (isB1){
                pGate = Mio_LibraryReadGateByName(pLib, "NR2D0BWP7T30P140HVT", nullptr); // Function = ~(A1+A2)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI21) " << pNode << " with new node " << pSub << " and new Fun NOR2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 2)); // Function = ~B
                SubId = GetId(pSub);
                cout << "replace (OAI21) " << pNode << " with its fanin (an Inv of B) " << pSub << endl;
                Replace(targetId, SubId);
            }
        }
    }
    else if (GetGateName(pNode).find("OA21D") != -1){   // Function = (A1+A2)B
        bool isB1 = IsConst1(GetFanin(pNode,2));
        bool isB0 = IsConst0(GetFanin(pNode,2));
        if (isC0){
            if (isB0){
                cout << "replace (OA21) " << pNode << " with const 0 " << endl;
                Replace(targetId, consts.first); //Function = 0
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "CKAN2D0BWP7T30P140HVT", nullptr); // Function = A1B
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OA21) " << pNode << " with new node " << pSub << " and new Fun AND2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
        else{
            if (isB1){
                pGate = Mio_LibraryReadGateByName(pLib, "OR2D0BWP7T30P140HVT", nullptr); // Function = A1+A2
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OA21) " << pNode << " with new node " << pSub << " and new Fun NOR2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                SubId = GetFaninId(targetId, 0);  // Function = B
                cout << "replace (OA21) " << pNode << " with its fanin (B) " << GetObj(SubId) << endl;
                Replace(targetId, SubId);
            }
        }
    }
    else if (GetGateName(pNode).find("OAI211D") != -1){ // Function = ~((A1+A2)BC)
        if (isC0){
            bool isA1 = IsConst0(GetFanin(pNode,0));
            bool isA2 = IsConst0(GetFanin(pNode,1));
            if(isA1 || isA2){
                // if(isA1)
                //     Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 1)); // Function = ~(BCA2)
                // else
                //     Abc_ObjAddFanin(pNewNode, GetFanin(pNode, 0)); // Function = ~(BCA1)
                pGate = Mio_LibraryReadGateByName(pLib, "ND3D0BWP7T30P140HVT", nullptr); // Function = NAND3 = ~(BCA1)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI211) " << pNode << " with new node " << pSub << " and new Fun NAND3" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                cout << "replace (OAI211) " << pNode << " with const 1 " << endl;
                Replace(targetId, consts.second); //Function = 1
            }
        }
        else{
            bool isA1 = IsConst1(GetFanin(pNode,0));
            bool isA2 = IsConst1(GetFanin(pNode,1));
            if(isA1 || isA2){
                pGate = Mio_LibraryReadGateByName(pLib, "CKND2D0BWP7T30P140HVT", nullptr); // Function = ~(BC)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI211) " << pNode << " with new node " << pSub << " and new Fun NAND2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "OAI21D1BWP7T30P140HVT", nullptr); // Function = ~((A1+A2)B)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OAI211) " << pNode << " with new node " << pSub << " and new Fun OAI21" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
    }
    else if (GetGateName(pNode).find("OA211D") != -1){  // Function = (A1+A2)BC
        if (isC0){
            bool isA1 = IsConst0(GetFanin(pNode,0));
            bool isA2 = IsConst0(GetFanin(pNode,1));
            if(isA1 || isA2){
                pGate = Mio_LibraryReadGateByName(pLib, "AN3D0BWP7T30P140HVT", nullptr); // Function = A1BC
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OA211) " << pNode << " with new node " << pSub << " and new Fun AND3" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                cout << "replace (OA211) " << pNode << " with const 0 " << endl;
                Replace(targetId, consts.first); //Function = 0
            }
        }
        else{
            bool isA1 = IsConst1(GetFanin(pNode,0));
            bool isA2 = IsConst1(GetFanin(pNode,1));
            if(isA1 || isA2){
                pGate = Mio_LibraryReadGateByName(pLib, "CKAN2D0BWP7T30P140HVT", nullptr); // Function = BC
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OA211) " << pNode << " with new node " << pSub << " and new Fun AND2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "OA21D1BWP7T30P140HVT", nullptr); // Function = (A1+A2)B
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (OA211) " << pNode << " with new node " << pSub << " and new Fun OA21" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
    }
    else if (GetGateName(pNode).find("XOR3D") != -1){   // Function = A^B^C XOR3D1BWP7T30P140HVT
        if (isC0)
            pGate = Mio_LibraryReadGateByName(pLib, "XOR2D0BWP7T30P140HVT", nullptr); // Function = XOR2
        else
            pGate = Mio_LibraryReadGateByName(pLib, "XNR2D0BWP7T30P140HVT", nullptr); // Function = XNOR2
        pNewNode->pData = pGate;
        pSub = pNewNode;
        std::string Fun = isC0? "XOR2" : "XNOR2";
        cout << "replace (XOR3) " << pNode << " with new node " << pSub << " and new Fun " << Fun << endl;
        TransfFanout(pNode, pSub);
        DelObj(pNode);
    }
    else if (GetGateName(pNode).find("XNR3D") != -1){   // Function = ~(A^B^C) XNR3D1BWP7T30P140HVT
        if (isC0)
            pGate = Mio_LibraryReadGateByName(pLib, "XNR2D0BWP7T30P140HVT", nullptr); // Function = XNOR2
        else
            pGate = Mio_LibraryReadGateByName(pLib, "XOR2D0BWP7T30P140HVT", nullptr); // Function = XOR2
        pNewNode->pData = pGate;
        pSub = pNewNode;
        std::string Fun = isC0? "XNOR2" : "XOR2";
        cout << "replace (XOR3) " << pNode << " with new node " << pSub << " and new Fun " << Fun << endl;
        TransfFanout(pNode, pSub);
        DelObj(pNode);
    }
    else if (GetGateName(pNode).find("CKND0B") != -1 || GetGateName(pNode).find("INVD") != -1){
        if (isC0){
            cout << "replace CKND(INV) " << pNode << " with const 1 " << endl;
            Replace(targetId, consts.second);
        }
        else{
            cout << "replace CKND(INV) " << pNode << " with const 0 " << endl;
            Replace(targetId, consts.first);
        }
    }
    else if (GetGateName(pNode).find("BUFFD") != -1 || GetGateName(pNode).find("CKBD") != -1){
        if (isC0){
            cout << "replace BUF " << pNode << " with const 0 " << endl;
            Replace(targetId, consts.first);
        }
        else{
            cout << "replace BUF " << pNode << " with const 1 " << endl;
            Replace(targetId, consts.second);
        }
    }
    else if (GetGateName(pNode).find("OR4D") != -1){    // Function = A + B + C + D
        if (isC0){
            pGate = Mio_LibraryReadGateByName(pLib, "OR3D0BWP7T30P140HVT", nullptr); // Function = A + B + C
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (OR4) " << pNode << " with new node " << pSub << " and new Fun OR3" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
        else{
            cout << "replace (OR4) " << pNode << " with const 1 " << endl;
            Replace(targetId, consts.second);
        }
    }
    else if (GetGateName(pNode).find("NR4D") != -1){    // Function = A + B + C + D
        if (isC0){
            pGate = Mio_LibraryReadGateByName(pLib, "NR3D0BWP7T30P140HVT", nullptr); // Function = A + B
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (NOR4) " << pNode << " with new node " << pSub << " and new Fun NOR3" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
        else{
            cout << "replace (NOR4) " << pNode << " with const 0 " << endl;
            Replace(targetId, consts.first);
        }
    }
    else if (GetGateName(pNode).find("OR3D") != -1){    // Function = A + B + C
        if (isC0){
            pGate = Mio_LibraryReadGateByName(pLib, "OR2D0BWP7T30P140HVT", nullptr); // Function = A + B
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (OR3) " << pNode << " with new node " << pSub << " and new Fun OR2" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
        else{
            cout << "replace (OR3) " << pNode << " with const 1 " << endl;
            Replace(targetId, consts.second);
        }
    }
    else if (GetGateName(pNode).find("NR3D") != -1){    // Function = ~(A + B + C) // POSSIBLY INR3 F = ~((~A) + B + C)
        if (GetGateName(pNode).find("INR3D") != -1){
            if (isC0){
                bool isA = IsConst0(GetFanin(pNode,0));
                if (!isA){
                    pGate = Mio_LibraryReadGateByName(pLib, "INR2D0BWP7T30P140HVT", nullptr); // Function = ~((~A) + B)
                    pNewNode->pData = pGate;
                    pSub = pNewNode;
                    cout << "replace (INOR3) " << pNode << " with new node " << pSub << " and new Fun INOR2" << endl;
                    TransfFanout(pNode, pSub);
                    DelObj(pNode);
                }
                else{
                    cout << "replace (INOR3) " << pNode << " with const 0 " << endl;
                    Replace(targetId, consts.first);
                }
            }
            else{
                bool isA = IsConst1(GetFanin(pNode,0));
                if (!isA){
                    cout << "replace (INOR3) " << pNode << " with const 0 " << endl;
                    Replace(targetId, consts.first);
                }
                else{
                    pGate = Mio_LibraryReadGateByName(pLib, "NR2D0BWP7T30P140HVT", nullptr); // Function = ~(B + C)
                    pNewNode->pData = pGate;
                    pSub = pNewNode;
                    cout << "replace (INOR3) " << pNode << " with new node " << pSub << " and new Fun NAND2" << endl;
                    TransfFanout(pNode, pSub);
                    DelObj(pNode);
                }
            }            
        }
        else{
            if (isC0){
                pGate = Mio_LibraryReadGateByName(pLib, "NR2D0BWP7T30P140HVT", nullptr); // Function = ~(A + B)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (NOR3) " << pNode << " with new node " << pSub << " and new Fun NOR2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                cout << "replace (NOR3) " << pNode << " with const 0 " << endl;
                Replace(targetId, consts.first);
            }
        }
    }
    else if (GetGateName(pNode).find("AN4D") != -1){    // Function = ABCD
        if (isC0){
            cout << "replace (AND4) " << pNode << " with const 0 " << endl;
            Replace(targetId, consts.first);
        }
        else{
            pGate = Mio_LibraryReadGateByName(pLib, "AN3D0BWP7T30P140HVT", nullptr); // Function = ABC
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (AND4) " << pNode << " with new node " << pSub << " and new Fun AND3" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
    }
    else if (GetGateName(pNode).find("ND4D") != -1){    // Function = ABC
        if (isC0){
            cout << "replace (NAND4) " << pNode << " with const 1 " << endl;
            Replace(targetId, consts.second);
        }
        else{
            pGate = Mio_LibraryReadGateByName(pLib, "ND3D0BWP7T30P140HVT", nullptr); // Function = AB
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (NAND4) " << pNode << " with new node " << pSub << " and new Fun NAND3" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
    }
    else if (GetGateName(pNode).find("AN3D") != -1){    // Function = ABC
        if (isC0){
            cout << "replace (AND3) " << pNode << " with const 0 " << endl;
            Replace(targetId, consts.first);
        }
        else{
            pGate = Mio_LibraryReadGateByName(pLib, "CKAN2D0BWP7T30P140HVT", nullptr); // Function = AB
            pNewNode->pData = pGate;
            pSub = pNewNode;
            cout << "replace (AND3) " << pNode << " with new node " << pSub << " and new Fun AND2" << endl;
            TransfFanout(pNode, pSub);
            DelObj(pNode);
        }
    }
    else if (GetGateName(pNode).find("ND3D") != -1){    // Function = ~(ABC)  // POSSIBLY IND3 F = ~((~A)BC)
        if (GetGateName(pNode).find("IND3D") != -1){
            if (isC0){
                bool isA = IsConst0(GetFanin(pNode,0));
                if (!isA){
                    cout << "replace (INAND3) " << pNode << " with const 1 " << endl;
                    Replace(targetId, consts.second);
                }
                else{
                    pGate = Mio_LibraryReadGateByName(pLib, "CKND2D0BWP7T30P140HVT", nullptr); // Function = ~(BC)
                    pNewNode->pData = pGate;
                    pSub = pNewNode;
                    cout << "replace (INAND3) " << pNode << " with new node " << pSub << " and new Fun NAND2" << endl;
                    TransfFanout(pNode, pSub);
                    DelObj(pNode);
                }
            }
            else{
                bool isA = IsConst1(GetFanin(pNode,0));
                if (!isA){
                    pGate = Mio_LibraryReadGateByName(pLib, "IND2D0BWP7T30P140HVT", nullptr); // Function = ~((~A)B)
                    pNewNode->pData = pGate;
                    pSub = pNewNode;
                    cout << "replace (INAND3) " << pNode << " with new node " << pSub << " and new Fun INAND2" << endl;
                    TransfFanout(pNode, pSub);
                    DelObj(pNode);
                }
                else{
                    cout << "replace (INAND3) " << pNode << " with const 1 " << endl;
                    Replace(targetId, consts.second);
                }
            }            
        }
        else{
            if (isC0){
                cout << "replace (NAND3) " << pNode << " with const 1 " << endl;
                Replace(targetId, consts.second);
            }
            else{
                pGate = Mio_LibraryReadGateByName(pLib, "CKND2D0BWP7T30P140HVT", nullptr); // Function = ~(AB)
                pNewNode->pData = pGate;
                pSub = pNewNode;
                cout << "replace (NAND3) " << pNode << " with new node " << pSub << " and new Fun NAND2" << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }   
        }
    }
    else if (GetGateName(pNode).find("OR2D") != -1){    // Function = A + B
        if (GetGateName(pNode).find("XOR2D") != -1){   // Function = A ^ B
            if (isC0){
                if (IsConst0(GetFaninId(targetId, 0))) //check whether the first input is 0
                    SubId = GetFaninId(targetId, 1); 
                else
                    SubId = GetFaninId(targetId, 0); 
                cout << "replace (XOR2) " << pNode << " with its fanin " << GetObj(SubId) << endl;
                Replace(targetId, SubId); // 0 ^ A = A
            }
            else{
                if (IsConst1(GetFaninId(targetId, 0)))
                    pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 1));
                else
                    pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 0));
                SubId = GetId(pSub);
                cout << "replace (XOR2) " << pNode << " with its fanin (an Inv of old fanin) " << pSub << endl;
                TransfFanout(pNode, pSub); // 1 ^ A = ~A
                DelObj(pNode);
            }
        }
        else{
            if (isC0){
                if (IsConst0(GetFaninId(targetId, 0))) //check whether the first input is 0
                    SubId = GetFaninId(targetId, 1); 
                else
                    SubId = GetFaninId(targetId, 0); 
                cout << "replace (OR2) " << pNode << " with its fanin " << GetObj(SubId) << endl;
                Replace(targetId, SubId); 
            }
            else{
                cout << "replace (OR2) " << pNode << " with const 1 " << endl;
                Replace(targetId, consts.second);
            }
        }
    }
    else if (GetGateName(pNode).find("NR2D") != -1){    // Function = ~(A + B) // POSSIBLY INR2 F = ~((~A) + B) and XNR2
        if (GetGateName(pNode).find("XNR2D") != -1){   // Function = ~(A ^ B)
            if (!isC0){
                if (IsConst1(GetFaninId(targetId, 0))) //check whether the first input is 1
                    SubId = GetFaninId(targetId, 1); 
                else
                    SubId = GetFaninId(targetId, 0); 
                cout << "replace (XNOR2) " << pNode << " with its fanin " << GetObj(SubId) << endl;
                Replace(targetId, SubId); // ~(1 ^ A) = A
            }
            else{
                if (IsConst0(GetFaninId(targetId, 0)))
                    pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 1));
                else
                    pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 0));
                SubId = GetId(pSub);
                cout << "replace (XNOR2) " << pNode << " with its fanin (an Inv of old fanin) " << pSub << endl;
                TransfFanout(pNode, pSub); // ~(0 ^ A) = ~A
                DelObj(pNode);
            }
        }
        else if (GetGateName(pNode).find("INR2D") != -1){
            bool isA = IsConst(GetFanin(pNode,0));
            if (!isA && isC0){
                SubId = GetFaninId(targetId, 0);  // Function = A
                cout << "replace (INOR2) " << pNode << " with its fanin " << GetObj(SubId) << endl;
                Replace(targetId, SubId);
            }
            else if (isA && !isC0){
                pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 1)); // Function = ~B
                SubId = GetId(pSub);
                cout << "replace (INOR2) " << pNode << " with its fanin (an Inv of old fanin) " << pSub << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                cout << "replace (INOR2) " << pNode << " with const 0 " << endl;
                Replace(targetId, consts.first);
            }
        }
        else{
            if (isC0){
                if (IsConst0(GetFaninId(targetId, 0)))
                    pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 1));
                else
                    pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 0));
                SubId = GetId(pSub);
                cout << "replace (NOR2) " << pNode << " with its fanin (an Inv of old fanin) " << pSub << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                cout << "replace (NOR2) " << pNode << " with const 0 " << endl;
                Replace(targetId, consts.first);
            }            
        }
    }
    else if (GetGateName(pNode).find("AN2D") != -1){    // Function = A B
        if (isC0){
            cout << "replace (AND2) " << pNode << " with const 0 " << endl;
            Replace(targetId, consts.first);
        }
        else{
            if (IsConst1(GetFaninId(targetId, 0))) //check whether the first input is 0
                SubId = GetFaninId(targetId, 1); 
            else
                SubId = GetFaninId(targetId, 0); 
            cout << "replace (AND2) " << pNode << " with its fanin " << GetObj(SubId) << endl;
            Replace(targetId, SubId); 
        }
    }
    else if (GetGateName(pNode).find("ND2D") != -1){    // Function = ~(A B) // POSSIBLY IND2 F = ~((~A)B)
        if (GetGateName(pNode).find("IND2D") != -1){
            bool isA0 = IsConst0(GetFanin(pNode,0));
            bool isA1 = IsConst1(GetFanin(pNode,0));
            if (!isA1 && !isC0){
                SubId = GetFaninId(targetId, 0);  // Function = A
                cout << "replace (INAND2) " << pNode << " with its fanin " << GetObj(SubId) << endl;
                Replace(targetId, SubId);
            }
            else if (isA0 && isC0){
                pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 1)); // Function = ~B
                SubId = GetId(pSub);
                cout << "replace (INAND2) " << pNode << " with its fanin (an Inv of old fanin) " << pSub << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
            else{
                cout << "replace (INAND2) " << pNode << " with const 1 " << endl;
                Replace(targetId, consts.second);
            }
        }
        else{
            if (isC0){
                cout << "replace (NAND2) " << pNode << " with const 1 " << endl;
                Replace(targetId, consts.second);
            }
            else{
                if (IsConst1(GetFaninId(targetId, 0)))
                    pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 1));
                else
                    pSub = Abc_NtkCreateNodeInv(GetNet(), GetFanin(targetId, 0));
                SubId = GetId(pSub);
                cout << "replace (NAND2) " << pNode << " with its fanin (an Inv of old fanin) " << pSub << endl;
                TransfFanout(pNode, pSub);
                DelObj(pNode);
            }
        }
    }
    else{
        cout << "cannot find a way to simplify " << pNode << " with " << GetGateName(pNode) << endl;
        assert(0);
    }
}

bool NetMan::ProcConstInp() {
    // special processing for Const Input
    if (GetNetType() != NET_TYPE::GATE)
        assert(0);
    bool isUpd = false;
    auto consts = CreateConst();
    ll count0 = 0;
    ll count1 = 0;
    for (auto i = 0; i < GetFanoutNum(consts.first); ++i){
        if (IsObjPo(GetFanoutId(consts.first, i)))
            count0++;
    }
    for (auto j = 0; j < GetFanoutNum(consts.second); ++j){
        if (IsObjPo(GetFanoutId(consts.second, j)))
            count1++;
    }
    cout << "there are " << count0 << " const 0, and " << count1 << " const 1 in POs, update the rest consts" << endl;
    while (GetFanoutNum(consts.first) > count0 || GetFanoutNum(consts.second) > count1) {
        if (GetFanoutNum(consts.first) > count0){
            auto targId = GetFanoutId(consts.first, count0);
            cout << "fixing const 0 : ";
            if (!IsNode(targId)){
                cout << "this node is no longer a node in the netlist, ignore it." << endl;
                count0++;
                continue;
            }
            if (IsObjPo(targId)){
                cout << "adding 1 PO with Const 0" << endl;
                count0++;
                consts = CreateConst();
                continue;
            }
            FixConstInp(consts.first, targId);
            CleanUp();
            consts = CreateConst();
            if (!isUpd)
                isUpd = true;
            continue;
        }
        if (GetFanoutNum(consts.second) > count1){
            cout << "fixing const 1 : ";
            auto targId = GetFanoutId(consts.second, count1);
            if (!IsNode(targId)){
                cout << "this node is no longer a node in the netlist, ignore it." << endl;
                count1++;
                continue;
            }
            if (IsObjPo(targId)){
                cout << "adding 1 PO with Const 1" << endl;
                count1++;
                consts = CreateConst();
                continue;
            }
            auto pNode = GetObj(targId);
            if (GetGateName(pNode).find("FA1") != -1 || GetGateName(pNode).find("HA1") != -1){
                bool hasConst0 = false;
                for (ll faninId = 0; faninId < GetFaninNum(pNode); ++faninId){
                    if (IsConst0(faninId))
                        hasConst0 = true;
                }
                if(hasConst0){
                    cout << "find FA/HA with both const 1 and 0, fix const 0 first..." << endl;
                    cout << "fixing const 0 : ";
                    FixConstInp(consts.first, targId);
                    consts = CreateConst();
                    if (!isUpd)
                        isUpd = true;
                    continue;
                }
            }
            FixConstInp(consts.second, targId);
            CleanUp();
            consts = CreateConst();
            if (!isUpd)
                isUpd = true;
        }
    }
    return isUpd;
}

pair<ll, ll> NetMan::HasDoubleInv(){
    pair <ll, ll> ret(-1, -1);
    if (GetNetType() != NET_TYPE::GATE)
        assert(0);
    abc::Abc_Obj_t* pObj = nullptr;
    int i = 0;
    Abc_NtkForEachNode(pNtk, pObj, i){
        if (GetGateName(pObj).find("CKND0BWP7T30P140HVT") == -1)
            continue;
        if (GetFanoutNum(pObj) != 1)
            continue;
        auto pFanout = GetFanout(pObj,0);
        if (GetGateName(pFanout).find("CKND0BWP7T30P140HVT") != -1 && !IsObjPo(pFanout)){
            ret.first = GetId(pObj);
            ret.second = GetId(pFanout);
            break;
        }
    }
    return ret;
}

bool NetMan::ProcDoubleInv(){
    if (GetNetType() != NET_TYPE::GATE)
        assert(0);
    bool isUpd = false;
    auto InvPair = HasDoubleInv();
    cout << InvPair.first << " and " << InvPair.second << endl;
    while(InvPair.first != -1 && InvPair.second != -1){
        cout << "find consecutive Inv pair (" << InvPair.first << ", " << InvPair.second << "): ";
        auto FaninId = GetFaninId(InvPair.first, 0);
        Replace(InvPair.second, FaninId);
        cout << "replace " << InvPair.second << " with node " << FaninId << endl;
        CleanUp();
        InvPair = HasDoubleInv();
        isUpd = true;
    }
    cout << "there is no consecutive Inv pair, end of this process" << endl;
    return isUpd;
}

bool NetMan::ProcHalfAndFullAdd() {
    // special processing for half/full adder
    if (GetNetType() != NET_TYPE::GATE)
        return false;
    bool isUpd = false;
    ll idMaxPlus1 = GetIdMaxPlus1();
    for (ll nodeId = 0; nodeId < idMaxPlus1; ++nodeId) {
        if (!IsNode(nodeId)) continue;
        auto pNode = GetObj(nodeId);
        if (GetGateName(pNode).find("HA1") != -1) {
            if (GetTwinNode(pNode) == nullptr) {
                cout << "cannot find twin for "; PrintObj(pNode, true);
                auto sop = string(Mio_GateReadSop((Mio_Gate_t *)pNode->pData));
                auto pLib = (Mio_Library_t *)Abc_FrameReadLibGen();
                auto pNewNode = Abc_NtkCreateNode(GetNet());
                for (ll faninId = 0; faninId < GetFaninNum(pNode); ++faninId)
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                Abc_Obj_t * pSub = nullptr;
                if (sop == "11 1\n") { // CO=A B
                    auto pGate = Mio_LibraryReadGateByName(pLib, "CKAN2D0BWP7T30P140HVT", nullptr);
                    assert(pGate != nullptr);
                    pNewNode->pData = pGate;
                    pSub = pNewNode;
                }
                else if (sop == "10 1\n01 1\n" || sop == "01 1\n10 1\n") { // S=A^B
                    auto pGate = Mio_LibraryReadGateByName(pLib, "XOR2D0BWP7T30P140HVT", nullptr);
                    assert(pGate != nullptr);
                    pNewNode->pData = pGate;
                    pSub = pNewNode;
                }
                else {
                    cout << sop;
                    assert(0);
                }
                assert(pSub != nullptr);
                cout << "replace " << pNode << " with new node " << pSub << endl;
                TransfFanout(pNode, pSub);
                Abc_ObjSetOriId(pSub, Abc_ObjGetOriId(pNode));  // update oriId
                isUpd = true;
            }
        }
        else if (GetGateName(pNode).find("FA1") != -1) {
            if (GetTwinNode(pNode) == nullptr) {
                cout << "cannot find twin for "; PrintObj(pNode, true);
                auto sop = string(Mio_GateReadSop((Mio_Gate_t *)pNode->pData));
                auto pLib = (Mio_Library_t *)Abc_FrameReadLibGen();
                auto pNewNode = Abc_NtkCreateNode(GetNet());
                for (ll faninId = 0; faninId < GetFaninNum(pNode); ++faninId)
                    Abc_ObjAddFanin(pNewNode, GetFanin(pNode, faninId));
                Abc_Obj_t * pSub = nullptr;
                if (sop == "1-1 1\n-11 1\n11- 1\n") { // CO=A B+B CI+A CI
                    auto pGate = Mio_LibraryReadGateByName(pLib, "MAOI222D0BWP7T30P140HVT", nullptr);
                    assert(pGate != nullptr);
                    pNewNode->pData = pGate;
                    pSub = Abc_NtkCreateNodeInv(GetNet(), pNewNode);
                }
                else if (sop == "100 1\n010 1\n111 1\n001 1\n") { // S=A^B^CI
                    auto pGate = Mio_LibraryReadGateByName(pLib, "XOR3D1BWP7T30P140HVT", nullptr);
                    assert(pGate != nullptr);
                    pNewNode->pData = pGate;
                    pSub = pNewNode;
                }
                else {
                    cout << sop;
                    assert(0);
                }
                assert(pSub != nullptr);
                cout << "replace " << pNode << " with new node " << pSub << endl;
                TransfFanout(pNode, pSub);
                Abc_ObjSetOriId(pSub, Abc_ObjGetOriId(pNode));  // update oriId
                Abc_ObjSetOriId(pNewNode, Abc_ObjGetOriId(pNode) * 10);
                isUpd = true;
            }
        }
    }
    return isUpd;
}


void NetMan::ProcHalfAndFullAddNew() {
    // special processing for half/full adder
    if (GetNetType() != NET_TYPE::GATE)
        return;
    unordered_set <ll> vis;
    vector <ll> targNodes;
    for (ll iNode = 0; iNode < GetIdMaxPlus1(); ++iNode) {
        if (!IsNode(iNode))
            continue;
        if (vis.count(iNode))
            continue;
        vis.emplace(iNode);
        auto pNode = GetObj(iNode);
        auto pTwin = GetTwinNode(pNode);
        if (pTwin == nullptr)
            continue;
        vis.emplace(pTwin->Id);
        if (GetGateName(pNode).find("HA1") != -1)
            targNodes.emplace_back(iNode);
        else if (GetGateName(pNode).find("FA1") != -1)
            targNodes.emplace_back(iNode);
        else
            assert(0);
    }

    for (ll targId: targNodes) {
        auto pNode = GetObj(targId);
        auto pTwin = GetTwinNode(pNode);
        assert(pTwin != nullptr);
        if (GetGateName(pNode).find("HA1") != -1) {
            // print
            // PrintObj(pNode, true); 
            auto sop0 = string(Mio_GateReadSop((Mio_Gate_t *)pNode->pData));
            // cout << sop0;
            // PrintObj(pTwin, true);
            auto sop1 = string(Mio_GateReadSop((Mio_Gate_t *)pTwin->pData));
            // cout << sop1;
            // cout << endl;

            // pNode sop0 S, pTwin sop1 CO
            assert(sop0 == "10 1\n01 1\n" && sop1 == "11 1\n");
            vector <Abc_Obj_t *> fanins;
            ll nFanin = GetFaninNum(pNode);
            assert(nFanin == GetFaninNum(pTwin) && nFanin == 2);
            for (ll iFanin = 0; iFanin < nFanin; ++iFanin) {
                assert(GetFanin(pNode, iFanin) == GetFanin(pTwin, iFanin));
                fanins.emplace_back(GetFanin(pNode, iFanin));
            }

            // create gates
            auto pNodeCo = CreateGate(std::vector <Abc_Obj_t *> ({fanins[0], fanins[1]}), "CKAN2D1BWP7T30P140HVT");
            auto pNodeN6 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[0], fanins[1]}), "NR2D0BWP7T30P140HVT");
            auto pNodeS = CreateGate(std::vector <Abc_Obj_t *> ({pNodeCo, pNodeN6}), "NR2D0BWP7T30P140HVT");

            // cout << "replace " << pTwin << " with new node " << pNodeCo << endl;
            TransfFanout(pTwin, pNodeCo);
            // cout << "replace " << pNode << " with new node " << pNodeS << endl;
            TransfFanout(pNode, pNodeS);
        }
        else if (GetGateName(pNode).find("FA1") != -1) {
            // print
            // PrintObj(pNode, true); 
            auto sop0 = string(Mio_GateReadSop((Mio_Gate_t *)pNode->pData));
            // cout << sop0;
            // PrintObj(pTwin, true);
            auto sop1 = string(Mio_GateReadSop((Mio_Gate_t *)pTwin->pData));
            // cout << sop1;
            // cout << endl;

            // pNode sop0 S, pTwin sop1 CO
            assert(sop0 == "100 1\n010 1\n111 1\n001 1\n" && sop1 == "1-1 1\n-11 1\n11- 1\n");
            vector <Abc_Obj_t *> fanins;
            ll nFanin = GetFaninNum(pNode);
            assert(nFanin == GetFaninNum(pTwin) && nFanin == 3);
            for (ll iFanin = 0; iFanin < nFanin; ++iFanin) {
                assert(GetFanin(pNode, iFanin) == GetFanin(pTwin, iFanin));
                fanins.emplace_back(GetFanin(pNode, iFanin));
            }

            // create gates
            auto pNodeN6 = CreateGate(std::vector <Abc_Obj_t *> ({fanins[1], fanins[2], fanins[1], fanins[2]}), "MOAI22D0BWP7T30P140HVT");
            auto pNodeS = CreateGate(std::vector <Abc_Obj_t *> ({fanins[0], pNodeN6, fanins[0], pNodeN6}), "MOAI22D0BWP7T30P140HVT");
            auto pNodeCo = CreateGate(std::vector <Abc_Obj_t *> ({fanins[1], fanins[2], fanins[0], pNodeN6}), "OA22D0BWP7T30P140HVT");

            // cout << "replace " << pTwin << " with new node " << pNodeCo << endl;
            TransfFanout(pTwin, pNodeCo);
            // cout << "replace " << pNode << " with new node " << pNodeS << endl;
            TransfFanout(pNode, pNodeS);
        }
    }
    CleanUp();
}

abc::Abc_Obj_t * NetMan::CreateGate(vector <Abc_Obj_t *> && fanins, const std::string & gateName) {
    auto pLib = (Mio_Library_t *)Abc_FrameReadLibGen();
    auto pGate = Mio_LibraryReadGateByName(pLib, const_cast <char *> (gateName.c_str()), nullptr);
    assert(pGate != nullptr);
    auto pNewNode = Abc_NtkCreateNode(GetNet());
    for (const auto & fanin: fanins)
        Abc_ObjAddFanin(pNewNode, fanin);
    pNewNode->pData = pGate;
    return pNewNode;
}

abc::Abc_Obj_t * NetMan::CreateGate2(vector <Abc_Obj_t *> && fanins, const std::string & gateName, const std::string & PoName) {
    auto pLib = (Mio_Library_t *)Abc_FrameReadLibGen();
    auto pGate = Mio_LibraryReadGateByName(pLib, const_cast <char *> (gateName.c_str()), const_cast <char *> (PoName.c_str()));
    assert(pGate != nullptr);
    auto pNewNode = Abc_NtkCreateNode(GetNet());
    for (const auto & fanin: fanins)
        Abc_ObjAddFanin(pNewNode, fanin);
    pNewNode->pData = pGate;
    return pNewNode;
}


std::string FixName(const char* name) {
    std::string newName(name);
    for (auto& ch: newName) {
        if (ch == '[' || ch == ']')
            ch = '_';
    }
    return newName;
}

void NetMan::DumpCFile(std::string&& fileName) {
    #ifdef DEBUG
    assert(isDupl == true);
    assert(this->GetNetType() == NET_TYPE::SOP || this->GetNetType() == NET_TYPE::GATE);
    #endif

    std::cout << "write " << fileName << endl;
    FILE * file = fopen(fileName.c_str(), "w");
    abc::Abc_Obj_t * pObj = nullptr;
    int i = 0;

    fprintf(file, "#include <stdbool.h>\n");
    fprintf(file, "void %s(", pNtk->pName);
    Abc_NtkForEachPi(pNtk, pObj, i)
        fprintf(file, "bool %s, ", FixName(Abc_ObjName(pObj)).c_str());
    Abc_NtkForEachPo(pNtk, pObj, i) {
        if (i != Abc_NtkPoNum(pNtk) - 1)
            fprintf(file, "bool& %s, ", FixName(Abc_ObjName(pObj)).c_str());
        else
            fprintf(file, "bool& %s)\n", FixName(Abc_ObjName(pObj)).c_str());
    }
    fprintf(file, "{\n");
    Abc_NtkForEachNode(pNtk, pObj, i) {
        std::ostringstream oss("");
        oss << "bool " << FixName(Abc_ObjName(pObj)).c_str() << " = ";

        if (Abc_NodeIsConst0(pObj)) {
            oss << "0;\n";
        }
        else if (Abc_NodeIsConst1(pObj)) {
            oss << "1;\n";
        }
        else {
            char *pSop = nullptr;
            if (this->GetNetType() == NET_TYPE::SOP)
                pSop = static_cast<char*>(pObj->pData);
            else if (this->GetNetType() == NET_TYPE::GATE)
                pSop = Mio_GateReadSop(static_cast<Mio_Gate_t*>(pObj->pData));
            else
                assert(0);
            int nVars = abc::Abc_SopGetVarNum(pSop);
            if (abc::Abc_SopIsComplement(pSop))
                oss << "!(\n";
            else
                oss << "(\n";
            for (char * pCube = pSop; *pCube; pCube += nVars + 3) {
                if (pCube == pSop)
                    oss << "( ";
                else
                    oss << "|| ( ";
                bool isFirst = true;
                for (int k = 0; pCube[k] != ' '; k++) {
                    abc::Abc_Obj_t * pFanin = Abc_ObjFanin(pObj, k);
                    std::string faninName = std::string(FixName(Abc_ObjName(pFanin)).c_str());
                    if (isFirst) {
                        if (pCube[k] == '0') {
                            isFirst = false;
                            oss << "!" << faninName << " ";
                        }
                        else if (pCube[k] == '1') {
                            isFirst = false;
                            oss << faninName << " ";
                        }
                        else if (pCube[k] == '-')
                            ;
                        else
                            assert(0);
                    }
                    else {
                        if (pCube[k] == '0')
                            oss << "&& !" << faninName << " ";
                        else if (pCube[k] == '1')
                            oss << "&& " << faninName << " ";
                        else if (pCube[k] == '-')
                            ;
                        else
                            assert(0);
                    }
                }
                oss << ")\n";
            }
            oss << ");\n";
        }
        fprintf(file, "%s", oss.str().c_str());
    }
    Abc_NtkForEachPo(pNtk, pObj, i) {
        abc::Abc_Obj_t * pDriver = Abc_ObjFanin0(pObj);
        // if (!Abc_ObjIsNode(pDriver)) {
        //     cout << "pObj is " << abc::Abc_ObjName(pObj) << ", pDriver is " << abc::Abc_ObjName(pDriver) << endl;
        //     assert(Abc_ObjIsNode(pDriver));
        // }
        assert(Abc_ObjIsNode(pDriver) || Abc_ObjIsPi(pDriver));
        fprintf(file, "%s = ", FixName(Abc_ObjName(pObj)).c_str());
        fprintf(file, "%s;\n", FixName(Abc_ObjName(pDriver)).c_str());
    }
    fprintf(file, "}\n");
    fclose(file);
}

void NetMan::SetAllOriIdAsId() {
    Abc_Obj_t * pObj;
    int i;
    Abc_NtkForEachObj(pNtk, pObj, i) {
        SetOriId(pObj, Abc_ObjId(pObj));
    }
}

ll NetMan::CalcMultiNodesMffcNum(std::vector<ll> nodesId) {
    Abc_NtkCleanMarkD();
    
    for (ll ii = 0; ii < nodesId.size(); ++ii) {
        Abc_Obj_t * pNode = GetObj(nodesId[ii]);
        pNode->fMarkD = 1;
    }

    Abc_Obj_t * pNode;
    ll i;
    Abc_NtkForEachNodeReverse(pNtk, pNode, i) {
        if (pNode->fMarkD == 0) {
            Abc_Obj_t * pFanout;
            ll j;
            bool flag = true;
            Abc_ObjForEachFanout(pNode, pFanout, j) {
                if (pFanout->fMarkD == 0) {
                    flag = false;
                    break;
                }
            }
            if (flag)
                pNode->fMarkD = 1;
        }
    }
    ll num = 0;
    Abc_NtkForEachNode(pNtk, pNode, i) {
        if (pNode->fMarkD == 1)
            ++num;
    }
    Abc_NtkCleanMarkD();
    return num;
}

void NetMan::Abc_NtkCleanMarkD() {
    Abc_Obj_t * pObj;
    int i;
    Abc_NtkForEachObj( pNtk, pObj, i )
        pObj->fMarkD = 0;
}

void NetMan::Abc_NtkCleanMarkDE() {
    Abc_Obj_t * pObj;
    int i;
    Abc_NtkForEachObj( pNtk, pObj, i ) {
        pObj->fMarkD = 0;
        pObj->fMarkE = 0;
    }
}

double NetMan::CalcMultiNodesMffcArea(std::vector<ll> nodesId) {
    Abc_NtkCleanMarkD();
    
    for (ll ii = 0; ii < nodesId.size(); ++ii) {
        Abc_Obj_t * pNode = GetObj(nodesId[ii]);
        pNode->fMarkD = 1;
    }

    Abc_Obj_t * pNode;
    ll i;
    Abc_NtkForEachNodeReverse(pNtk, pNode, i) {
        if (Abc_NodeIsConst(pNode))
            continue;
        if (pNode->fMarkD == 0) {
            Abc_Obj_t * pFanout;
            ll j;
            bool flag = true;
            Abc_ObjForEachFanout(pNode, pFanout, j) {
                if (pFanout->fMarkD == 0) {
                    flag = false;
                    break;
                }
            }
            if (flag)
                pNode->fMarkD = 1;
        }
    }
    double area = 0;
    if (GetNetType() == NET_TYPE::GATE) {
        Abc_NtkForEachNode(pNtk, pNode, i) {
            if (pNode->fMarkD == 1) {
                string sop = string(Mio_GateReadSop((Mio_Gate_t *)pNode->pData));
                auto pLib = (Mio_Library_t *)Abc_FrameReadLibGen();
                if (GetGateName(pNode).find("HA1") != string::npos) {
                    if (sop == "11 1\n") { // CO=A B (AND)
                        Mio_Gate_t * pGate = Mio_LibraryReadGateByName(pLib, "XOR2D0BWP7T30P140HVT", nullptr);
                        assert(pGate != nullptr);
                        area += Mio_GateReadArea((Mio_Gate_t *)pNode->pData) - Mio_GateReadArea(pGate);
                    }
                    else if (sop == "10 1\n01 1\n" || sop == "01 1\n10 1\n") { // S=A^B (XOR2)
                        Mio_Gate_t * pGate = Mio_LibraryReadGateByName(pLib, "CKAN2D0BWP7T30P140HVT", nullptr);
                        assert(pGate != nullptr);
                        area += Mio_GateReadArea((Mio_Gate_t *)pNode->pData) - Mio_GateReadArea(pGate);
                    }
                }
                else if (GetGateName(pNode).find("FA1") != string::npos) {
                    if (sop == "1-1 1\n-11 1\n11- 1\n") { // CO=A B+B CI+A CI (MAOI)
                        Mio_Gate_t * pGate = Mio_LibraryReadGateByName(pLib, "XOR3D1BWP7T30P140HVT", nullptr);
                        assert(pGate != nullptr);
                        area += Mio_GateReadArea((Mio_Gate_t *)pNode->pData) - Mio_GateReadArea(pGate);
                    }
                    else if (sop == "100 1\n010 1\n111 1\n001 1\n") { // S=A^B^CI (XOR3)
                        Mio_Gate_t * pGate = Mio_LibraryReadGateByName(pLib, "MAOI222D0BWP7T30P140HVT", nullptr);
                        Mio_Gate_t * pInv = Mio_LibraryReadGateByName(pLib, "INVD1BWP7T30P140HVT", nullptr);
                        assert(pGate != nullptr);
                        area += Mio_GateReadArea((Mio_Gate_t *)pNode->pData) - Mio_GateReadArea(pGate) - Mio_GateReadArea(pInv);
                    }
                }
                else
                    area += Mio_GateReadArea((Mio_Gate_t *)pNode->pData);
            }
        }
    }
    else 
        area = -1;
    // Abc_NtkCleanMarkD();
    return area;
}

double NetMan::CalcMarkArea() {
    double area = 0;
    assert(GetNetType() == NET_TYPE::GATE);
    Abc_Obj_t * pNode;
    ll i;
    Abc_NtkForEachNode(pNtk, pNode, i) {
        if (pNode->fMarkD == 1) {
            string sop = string(Mio_GateReadSop((Mio_Gate_t *)pNode->pData));
            auto pLib = (Mio_Library_t *)Abc_FrameReadLibGen();
            if (GetGateName(pNode).find("HA1") != string::npos) {
                Abc_Obj_t * pTwin = GetTwinNode(pNode);
                if (pTwin != nullptr) {
                    if (pTwin->fMarkD == 1) {
                        area += Mio_GateReadArea((Mio_Gate_t *)pNode->pData);
                        ++i;
                        continue;
                    }
                }
                else {
                    if (sop == "11 1\n") { // CO=A B (AND)
                        Mio_Gate_t * pGate = Mio_LibraryReadGateByName(pLib, "XOR2D0BWP7T30P140HVT", nullptr);
                        assert(pGate != nullptr);
                        area += Mio_GateReadArea((Mio_Gate_t *)pNode->pData) - Mio_GateReadArea(pGate);
                    }
                    else if (sop == "10 1\n01 1\n" || sop == "01 1\n10 1\n") { // S=A^B (XOR2)
                        Mio_Gate_t * pGate = Mio_LibraryReadGateByName(pLib, "CKAN2D0BWP7T30P140HVT", nullptr);
                        assert(pGate != nullptr);
                        area += Mio_GateReadArea((Mio_Gate_t *)pNode->pData) - Mio_GateReadArea(pGate);
                    }
                }
            }
            else if (GetGateName(pNode).find("FA1") != string::npos) {
                Abc_Obj_t * pTwin = GetTwinNode(pNode);
                if (pTwin != nullptr) {
                    if (pTwin->fMarkD == 1) {
                        area += Mio_GateReadArea((Mio_Gate_t *)pNode->pData);
                        ++i;
                        continue;
                    }
                }
                else {
                    if (sop == "1-1 1\n-11 1\n11- 1\n") { // CO=A B+B CI+A CI (MAOI)
                        Mio_Gate_t * pGate = Mio_LibraryReadGateByName(pLib, "XOR3D1BWP7T30P140HVT", nullptr);
                        assert(pGate != nullptr);
                        area += Mio_GateReadArea((Mio_Gate_t *)pNode->pData) - Mio_GateReadArea(pGate);
                    }
                    else if (sop == "100 1\n010 1\n111 1\n001 1\n") { // S=A^B^CI (XOR3)
                        Mio_Gate_t * pGate = Mio_LibraryReadGateByName(pLib, "MAOI222D0BWP7T30P140HVT", nullptr);
                        Mio_Gate_t * pInv = Mio_LibraryReadGateByName(pLib, "INVD1BWP7T30P140HVT", nullptr);
                        assert(pGate != nullptr);
                        area += Mio_GateReadArea((Mio_Gate_t *)pNode->pData) - Mio_GateReadArea(pGate) - Mio_GateReadArea(pInv);
                    }
                }
            }
            else
                area += Mio_GateReadArea((Mio_Gate_t *)pNode->pData);
        }
    }
    return area;
}

void NetMan::CleanTravIds() {
    pNtk->nTravIds = 0;
    Abc_Obj_t * pNode;
    int i;
    Abc_NtkForEachNode(pNtk, pNode, i) {
        Abc_NodeSetTravId(pNode, 0);
    }
}

double NetMan::GetNodeArea(Abc_Obj_t * pNode) {
    assert(Abc_NtkIsMappedLogic(pNtk));
    double area = 0;

    string sop = string(Mio_GateReadSop((Mio_Gate_t *)pNode->pData));
    auto pLib = (Mio_Library_t *)Abc_FrameReadLibGen();
    if (GetGateName(pNode).find("HA1") != string::npos) {
        if (sop == "11 1\n") { // CO=A B (AND)
            Mio_Gate_t * pGate = Mio_LibraryReadGateByName(pLib, "XOR2D0BWP7T30P140HVT", nullptr);
            assert(pGate != nullptr);
            area = Mio_GateReadArea((Mio_Gate_t *)pNode->pData) - Mio_GateReadArea(pGate);
        }
        else if (sop == "10 1\n01 1\n" || sop == "01 1\n10 1\n") { // S=A^B (XOR2)
            Mio_Gate_t * pGate = Mio_LibraryReadGateByName(pLib, "CKAN2D0BWP7T30P140HVT", nullptr);
            assert(pGate != nullptr);
            area = Mio_GateReadArea((Mio_Gate_t *)pNode->pData) - Mio_GateReadArea(pGate);
        }
    }
    else if (GetGateName(pNode).find("FA1") != string::npos) {
        if (sop == "1-1 1\n-11 1\n11- 1\n") { // CO=A B+B CI+A CI (MAOI)
            Mio_Gate_t * pGate = Mio_LibraryReadGateByName(pLib, "XOR3D1BWP7T30P140HVT", nullptr);
            assert(pGate != nullptr);
            area = Mio_GateReadArea((Mio_Gate_t *)pNode->pData) - Mio_GateReadArea(pGate);
        }
        else if (sop == "100 1\n010 1\n111 1\n001 1\n") { // S=A^B^CI (XOR3)
            Mio_Gate_t * pGate = Mio_LibraryReadGateByName(pLib, "MAOI222D0BWP7T30P140HVT", nullptr);
            Mio_Gate_t * pInv = Mio_LibraryReadGateByName(pLib, "INVD1BWP7T30P140HVT", nullptr);
            assert(pGate != nullptr);
            area = Mio_GateReadArea((Mio_Gate_t *)pNode->pData) - Mio_GateReadArea(pGate) - Mio_GateReadArea(pInv);
        }
    }
    else
        area = Mio_GateReadArea((Mio_Gate_t *)pNode->pData);
    
    return area;
}

void NetMan::CheckDivisorNumForDisjNodes() {
    bool fConsiderOnlyDisjoint = true;
    bool fPrintOnly4LI = true;
    if (fConsiderOnlyDisjoint)
        cout << endl << "Restrict LOs to be disjoint" << endl;
    else
        cout << endl << "Do NOT restrict LOs to be disjoint" << endl;

    // get the influenced PO set for each node
    std::vector < boost::dynamic_bitset <ull> > poMarks;
    poMarks.resize(GetIdMaxPlus1(), boost::dynamic_bitset <ull>(GetPoNum(), 0));
    for (ll i = 0; i < GetIdMaxPlus1(); ++i) {
        if (GetObj(i) == nullptr)
            continue;
        poMarks[i].reset();
    }

    for (ll i = 0; i < GetPoNum(); ++i)
        poMarks[GetPoId(i)].set(i);
    Abc_Obj_t * pObj;
    ll i;
    Abc_NtkForEachObjReverse(pNtk, pObj, i) {
        if (pObj == nullptr)
            continue;
        ll i = GetId(pObj);
        for (ll j = 0; j < GetFanoutNum(pObj); ++j)
            poMarks[i] |= poMarks[GetFanoutId(pObj, j)];
    }

    // find 2 nodes as LO, then check MFFW
    cout << "find 2 nodes as LO" << endl;
    std::unordered_map<ll, std::unordered_map<ll, ll>> stats2;
    for (ll i = 0; i < GetIdMaxPlus1(); ++i) {
        Abc_Obj_t * pObj1 = GetObj(i);
        if (pObj1 == nullptr)
            continue;
        if (!Abc_ObjIsNode(pObj1))  // exclude PI and PO
            continue;
        if (Abc_NodeIsConst(pObj1)) // exclude CONST nodes
            continue;
        for (ll j = i + 1; j < GetIdMaxPlus1(); ++j) {
            Abc_Obj_t * pObj2 = GetObj(j);
            if (pObj2 == nullptr)
                continue;           
            if (!Abc_ObjIsNode(pObj2))  // exclude PI and PO
                continue;
            if (Abc_NodeIsConst(pObj2)) // exclude CONST nodes
                continue;
            if (fConsiderOnlyDisjoint) {
                if (!CheckDisjointness(poMarks[i], poMarks[j])) {
                    // cout << "i = " << i << ", j = " << j << endl;
                    continue;
                }       
            }        
            Abc_NtkCleanMarkDE();
            
            // fMarkD = 1: is mffc; fMarkE = 1: have been expanded/explored
            pObj1->fMarkD = 1;
            pObj2->fMarkD = 1;
            set <ll> LIs;
            Abc_Obj_t * pFanin;
            ll k;
            Abc_ObjForEachFanin(pObj1, pFanin, k) {
                LIs.insert(pFanin->Id);
            }
            Abc_ObjForEachFanin(pObj2, pFanin, k) {
                LIs.insert(pFanin->Id);
            }
            while (1) {
                set <ll> LIsNew;
                Abc_Obj_t * pExpandNode;
                ll expandId = 0;
                // select a node to expand forward (the direction to PI)
                for (auto it = LIs.rbegin(); it != LIs.rend(); ++it) {  // topo order (from largest ID to smallest ID)
                    expandId = *it;
                    pExpandNode = GetObj(expandId);
                    if (!pExpandNode->fMarkE)
                        break;
                    else
                        expandId = 0;
                }
                if (expandId == 0)
                    break;

                // expand
                if (Abc_ObjIsPi(pExpandNode)) {
                    pExpandNode->fMarkE = 1;
                    continue;
                }
                else {
                    bool fCanExpand = true;
                    Abc_Obj_t * pFanout;
                    Abc_ObjForEachFanout(pExpandNode, pFanout, k) {
                        if (!pFanout->fMarkD) {
                            fCanExpand = false;
                            break;
                        }
                    }

                    pExpandNode->fMarkE = 1;    // mark: have been explored
                    if (fCanExpand) {
                        // mark as in MFFC
                        pExpandNode->fMarkD = 1;
                        // remove the expanded node
                        auto it = LIs.find(expandId);
                        assert(it != LIs.end());
                        LIs.erase(it);
                        // add the fanins of the expanded node
                        Abc_ObjForEachFanin(pExpandNode, pFanin, k) {
                            LIs.insert(pFanin->Id);
                        }

                        // check FFW
                        if ((!fPrintOnly4LI) || (LIs.size() <= 4)) {
                            ll nMffc = 0;
                            Abc_NtkForEachNode(pNtk, pObj, k) {
                                if (pObj->fMarkD)
                                    ++nMffc;
                            }
                            if (nMffc > 2) {
                                stats2[nMffc][LIs.size()]++;
                                // cout << "LO(" << i << ", " << j << "), LI(";
                                // for (ll iLI : LIs) {
                                //     cout << iLI << ", ";
                                // }
                                // cout << "), #mffc = " << nMffc << endl;
                            }
                        }
                    }
                }
            }
            
            // ll nMffc = 0;
            // Abc_NtkForEachNode(pNtk, pObj, k) {
            //     if (pObj->fMarkD)
            //         ++nMffc;
            // }
            // if (nMffc > 2) {
            //     stats2[nMffc][LIs.size()]++;
            //     cout << "LO: i = " << i << ", j = " << j << ": #LI = " << LIs.size() << ", #mffc = " << nMffc << endl;
            // }
        }
    }
    cout << endl << "results for 2 LOs: " << endl;
    cout << "nMffc\t nLi\t nCircuit\n";
    for (const auto& [nMffc, liMap] : stats2) {
        for (const auto& [nLi, nCircuit] : liMap) {
            std::cout << std::setw(6) << nMffc << "\t"
                      << std::setw(4) << nLi << "\t"
                      << std::setw(9) << nCircuit << "\n";
        }
    }

    // find 3 nodes as LO, then check MFFW
    cout << endl << "find 3 nodes as LO" << endl;
    std::unordered_map<ll, std::unordered_map<ll, ll>> stats3;
    for (ll i = 0; i < GetIdMaxPlus1(); ++i) {
        Abc_Obj_t * pObj1 = GetObj(i);
        if (pObj1 == nullptr)
            continue;
        if (!Abc_ObjIsNode(pObj1))  // exclude PI and PO
            continue;
        if (Abc_NodeIsConst(pObj1)) // exclude CONST nodes
            continue;
        for (ll j = i + 1; j < GetIdMaxPlus1(); ++j) {
            Abc_Obj_t * pObj2 = GetObj(j);
            if (pObj2 == nullptr)
                continue;           
            if (!Abc_ObjIsNode(pObj2))  // exclude PI and PO
                continue;
            if (Abc_NodeIsConst(pObj2)) // exclude CONST nodes
                continue;
            if (fConsiderOnlyDisjoint) {    
                if (!CheckDisjointness(poMarks[i], poMarks[j]))
                    continue;
            }
            for (ll l = j + 1; l < GetIdMaxPlus1(); ++l) {
                Abc_Obj_t * pObj3 = GetObj(l);
                if (pObj3 == nullptr)
                    continue;           
                if (!Abc_ObjIsNode(pObj3))  // exclude PI and PO
                    continue;
                if (Abc_NodeIsConst(pObj3)) // exclude CONST nodes
                    continue;
                if (fConsiderOnlyDisjoint) {
                    if ((!CheckDisjointness(poMarks[l], poMarks[i])) || (!CheckDisjointness(poMarks[l], poMarks[j])))
                        continue;
                }

                Abc_NtkCleanMarkDE();
        
                // fMarkD = 1: is mffc; fMarkE = 1: have been expanded/explored
                pObj1->fMarkD = 1;
                pObj2->fMarkD = 1;
                pObj3->fMarkD = 1;
                set <ll> LIs;
                Abc_Obj_t * pFanin;
                ll k;
                Abc_ObjForEachFanin(pObj1, pFanin, k) {
                    LIs.insert(pFanin->Id);
                }
                Abc_ObjForEachFanin(pObj2, pFanin, k) {
                    LIs.insert(pFanin->Id);
                }
                Abc_ObjForEachFanin(pObj3, pFanin, k) {
                    LIs.insert(pFanin->Id);
                }
                while (1) {
                    set <ll> LIsNew;
                    Abc_Obj_t * pExpandNode;
                    ll expandId = 0;
                    // select a node to expand forward (the direction to PI)
                    for (auto it = LIs.rbegin(); it != LIs.rend(); ++it) {  // topo order (from largest ID to smallest ID)
                        expandId = *it;
                        pExpandNode = GetObj(expandId);
                        if (!pExpandNode->fMarkE)
                            break;
                        else
                            expandId = 0;
                    }
                    if (expandId == 0)
                        break;
    
                    // expand
                    if (Abc_ObjIsPi(pExpandNode)) {
                        pExpandNode->fMarkE = 1;
                        continue;
                    }
                    else {
                        bool fCanExpand = true;
                        Abc_Obj_t * pFanout;
                        Abc_ObjForEachFanout(pExpandNode, pFanout, k) {
                            if (!pFanout->fMarkD) {
                                fCanExpand = false;
                                break;
                            }
                        }
    
                        pExpandNode->fMarkE = 1;    // mark: have been explored
                        if (fCanExpand) {
                            // mark as in MFFC
                            pExpandNode->fMarkD = 1;
                            // remove the expanded node
                            auto it = LIs.find(expandId);
                            assert(it != LIs.end());
                            LIs.erase(it);
                            // add the fanins of the expanded node
                            Abc_ObjForEachFanin(pExpandNode, pFanin, k) {
                                LIs.insert(pFanin->Id);
                            }

                            // check FFW
                            if ((!fPrintOnly4LI) || (LIs.size() <= 4)) {
                                ll nMffc = 0;
                                Abc_NtkForEachNode(pNtk, pObj, k) {
                                    if (pObj->fMarkD)
                                        ++nMffc;
                                }
                                if (nMffc > 3) {
                                    stats3[nMffc][LIs.size()]++;
                                    // cout << "LO(" << i << ", " << j << ", " << l << "), LI(";
                                    // for (ll iLI : LIs) {
                                    //     cout << iLI << ", ";
                                    // }
                                    // cout << "), #mffc = " << nMffc << endl;
                                }
                            }
                        }
                    }
                }

                // ll nMffc = 0;
                // Abc_NtkForEachNode(pNtk, pObj, k) {
                //     if (pObj->fMarkD)
                //         ++nMffc;
                // }
                // if (nMffc > 3) {
                //     stats3[nMffc][LIs.size()]++;
                //     cout << "LO: i = " << i << ", j = " << j << ", l = " << l << ": #LI = " << LIs.size() << ", #mffc = " << nMffc << endl;
                // }                   
            }
        }
    }
    cout << endl << "results for 3 LOs: " << endl;
    cout << "nMffc\t nLi\t nCircuit\n";
    for (const auto& [nMffc, liMap] : stats3) {
        for (const auto& [nLi, nCircuit] : liMap) {
            cout << std::setw(6) << nMffc << "\t"
                    << std::setw(4) << nLi << "\t"
                    << std::setw(9) << nCircuit << "\n";
        }
    }
    cout << endl << endl;

    for (ll i = 0; i < GetIdMaxPlus1(); ++i) {
        cout << "poMarks[" << i << "] = " << poMarks[i] << endl;
    }
    cout << endl;
}

bool NetMan::CheckDisjointness(boost::dynamic_bitset <ull> poMark1, boost::dynamic_bitset <ull> poMark2) {
    for (ll i = 0; i < GetPoNum(); ++i) {
        if (poMark1[i] && poMark2[i])   // overlap
            return 0;
    }
    return 1;
}

bool AreAllVectorElements0s(vector <int> vec) {
    for (ll i = 0; i < vec.size(); ++i) {
        if (vec[i] != 0)
            return 0;
    }
    return 1;
}

bool NetMan::IsPathExist3(ll id1, ll id2, ll id3) {
    // check topological order
    assert(id1 < id2);
    assert(id2 < id3);
    return (IsPathExist2(id1, id2) || IsPathExist2(id1, id3) || IsPathExist2(id2, id3));
}

bool NetMan::IsPathExist2(ll id1, ll id2) {
    assert(id1 < id2);
    SetNetNotTrav();

    if (GetObjLev(id1) > 0.5 * level) {
        for (ll i = 0; i < GetFanoutNum(id1); ++i) {
            auto pFanout = GetFanout(id1, i);
            if (!GetObjTrav(pFanout)) {
                if (pFanout->Id == id2)
                    return 1;  
                if (SearchTFORec(id2, pFanout))
                    return 1;
            }
        }
        return 0;
    }  
    else {
        for (ll i = 0; i < GetFaninNum(id2); ++i) {
            auto pFanin = GetFanin(id2, i);
            if (!GetObjTrav(pFanin)) {
                if (pFanin->Id == id1)
                    return 1;  
                if (SearchTFIRec(id1, pFanin))
                    return 1;
            }
        }
        return 0;
    } 
}

bool NetMan::SearchTFORec(ll targId, abc::Abc_Obj_t * pObj) const {
    if (!IsNode(pObj))
        return 0;
    SetObjTrav(pObj);
    for (ll i = 0; i < GetFanoutNum(pObj); ++i) {
        auto pFanout = GetFanout(pObj, i);
        if (!GetObjTrav(pFanout)) {
            if (pFanout->Id == targId)
                return 1;
            if (SearchTFORec(targId, pFanout))
                return 1;
        }
    }
    return 0;
}

bool NetMan::SearchTFIRec(ll targId, abc::Abc_Obj_t * pObj) const {
    if (!IsNode(pObj))
        return 0;
    SetObjTrav(pObj);
    for (ll i = 0; i < GetFaninNum(pObj); ++i) {
        auto pFanin = GetFanin(pObj, i);
        if (!GetObjTrav(pFanin)) {
            if (pFanin->Id == targId)
                return 1;
            if (SearchTFIRec(targId, pFanin))
                return 1;
        }
    }
    return 0;
}

double NetMan::GetInvArea() const {
    auto pLib = (Mio_Library_t *)Abc_FrameReadLibGen();
    Mio_Gate_t * pInv = Mio_LibraryReadGateByName(pLib, "INVD1BWP7T30P140HVT", nullptr);
    assert(pInv != nullptr);
    return Mio_GateReadArea(pInv);
}

void NetMan::ReplaceSubCkt(vector <ll> vDiv, vector <ll> vLO, Abc_Ntk_t * pSubNtk, vector <ll> LoIndex) {
    Abc_Obj_t * pNode, * pFanout, * pFanin;
    ll i, j;
    // duplicate the nodes
    Abc_NtkCleanCopy(pSubNtk);
    Abc_NtkForEachNodeReverse(pSubNtk, pNode, i) {
        Abc_NtkDupObj(pNtk, pNode, 0);  // copy node to current pNtk
    }

    // connect all objects
    ll firstPoId = Abc_ObjId(Abc_NtkPo(pSubNtk, 0));
    Abc_NtkForEachNodeReverse(pSubNtk, pNode, i) {
        if (Abc_ObjIsPo(Abc_ObjFanout0(pNode))) {     // PO driver
            Abc_Obj_t * pPo = Abc_ObjFanout0(pNode);
            ll iPo = pPo->Id - firstPoId;

            vector <ll> connectedLoIds;
            for (ll j = 0; j < LoIndex.size(); ++j) {
                if (LoIndex[j] == iPo)
                    connectedLoIds.push_back(j);
            }
            assert(!connectedLoIds.empty());

            for (const auto & LoId : connectedLoIds) {
                Abc_Obj_t * pLO = GetObj(vLO[LoId]);
                Vec_Ptr_t * vObjFanouts = Vec_PtrAlloc(Abc_ObjFanoutNum(pLO));
                Abc_NodeCollectFanouts(pLO, vObjFanouts);
                Vec_PtrForEachEntry(Abc_Obj_t *, vObjFanouts, pFanout, j)
                    Abc_ObjPatchFanin(pFanout, pLO, pNode->pCopy);
                Vec_PtrFree(vObjFanouts);
            }
        }
        
        Abc_ObjForEachFanin(pNode, pFanin, j){
            if (!Abc_ObjIsPi(pFanin))
                Abc_ObjAddFanin(pNode->pCopy, pFanin->pCopy);
            else {
                for (int k = 0; k < vDiv.size(); k++)
                    if (pFanin == Abc_NtkPi(pSubNtk, k)) 
                        Abc_ObjAddFanin(pNode->pCopy, Abc_NtkObj(pNtk, vDiv[k]));
            }
        }
    }

    // delete original sub-circuit
    // for (ll o = 0; o < vLO.size(); ++o) {
    for (ll o = vLO.size() - 1; o >= 0; --o) {      // take fRelation == 1 case into account 
        Abc_Obj_t * pLO = GetObj(vLO[o]);
        if (Abc_ObjFanoutNum(pLO) == 0) {
            Abc_NtkDeleteObj_rec(pLO, 1);
        }
        else {
            cout << "#fanout = " << Abc_ObjFanoutNum(pLO) << endl;
            cout << "LO(n" << vLO[o] << ")'s fanout: " << endl;
            Abc_ObjForEachFanout(pLO, pFanout, i) {
                cout << pFanout->Id << " ";
            }
            cout << endl;
            PrintPro(1, 1, 0);
            assert(0);
        }
    }
}

void Abc_MfsWinSweepLeafTfo_rec_Pro(Abc_Obj_t * pObj)
{
    Abc_Obj_t * pFanout;
    int i;
    if ( Abc_ObjIsCo(pObj) )
        return;
    // if ( Abc_NodeIsTravIdCurrent(pObj) )
    if (pObj->fMarkA == 1)
        return;
    // Abc_NodeSetTravIdCurrent( pObj );
    pObj->fMarkA = 1;
    Abc_ObjForEachFanout( pObj, pFanout, i )
        Abc_MfsWinSweepLeafTfo_rec_Pro( pFanout );
}

void PrintNtk(Abc_Ntk_t * pNtk) {
    assert(Abc_NtkIsMappedLogic(pNtk));
    Abc_Obj_t * pObj, * pFanin, * pFanout;
    int i, j;
    std::cout << "total area = " << Abc_NtkGetMappedArea(pNtk) << endl;
    Abc_NtkForEachObj(pNtk, pObj, i) {
        if (Abc_ObjIsNode(pObj)) {
            std::cout << "Node id = " << Abc_ObjId(pObj) << " (" << std::string(abc::Mio_GateReadName(static_cast <abc::Mio_Gate_t *> (pObj->pData))) << ") ";
            std::cout << "Obj name: " << Abc_ObjName(pObj) << " fanins: ";
            Abc_ObjForEachFanin(pObj, pFanin, j) {
                std::cout << Abc_ObjId(pFanin) << " ";
            }
            std::cout << "fanout(" << Abc_ObjFanoutNum(pObj) << "): ";
            Abc_ObjForEachFanout(pObj, pFanout, j) {
                std::cout << Abc_ObjId(pFanout) << " ";
            }
            // area
            double pObjArea = Mio_GateReadArea((static_cast <Mio_Gate_t *> (pObj->pData)));
            std::cout << "area = " << pObjArea;
            // level
            std::cout << " level = " << pObj->Level;
            std::cout << endl;
        }
        else if (Abc_ObjIsPi(pObj)) {
            std::cout << "Node id = " << Abc_ObjId(pObj) << "(PI) ";
            std::cout << "Obj name: " << Abc_ObjName(pObj) << " fanout(" << Abc_ObjFanoutNum(pObj) << "): ";
            Abc_ObjForEachFanout(pObj, pFanout, j) {
                std::cout << Abc_ObjId(pFanout) << " ";
            }
            // level
            std::cout << " level = " << pObj->Level;
            std::cout << endl;
        }
        else if (Abc_ObjIsPo(pObj)) {
            std::cout << "Node id = " << Abc_ObjId(pObj) << "(PO) ";
            std::cout << "Obj name: " << Abc_ObjName(pObj) << " fanin: ";
            Abc_ObjForEachFanin(pObj, pFanin, j) {
                std::cout << Abc_ObjId(pFanin) << " ";
            }
            // level
            std::cout << " level = " << pObj->Level;
            std::cout << endl;
        }
    } 
    // std::cout << endl;
}

ll CountIntersection(const set<ll>& A, const set<ll>& B) {
    ll count = 0;
    auto it1 = A.begin();
    auto it2 = B.begin();
    while (it1 != A.end() && it2 != B.end()) {
        if (*it1 == *it2) {
            ++count;
            ++it1;
            ++it2;
        } else if (*it1 < *it2) {
            ++it1;
        } else {
            ++it2;
        }
    }
    return count;
}

bool NetMan::IsAllNodeMarkA1() const {
    Abc_Obj_t * pNode;
    int i;
    Abc_NtkForEachNode(pNtk, pNode, i) {
        if (pNode->fMarkA != 1)
            return 0;
    }
    return 1;
}


ll NetMan::GetNewId(ll oriId) const {
    if (newIdMap.find(oriId) == newIdMap.end()) {
        // not found in newIdMap
        if (oriId >= GetIdMaxPlus1()) {
            cout << "oriId = " << oriId << " is greater than GetIdMaxPlus1() = " << GetIdMaxPlus1() << endl;
            assert(0);
        }
        return oriId;
    }
    else
        return newIdMap.at(oriId);
}

void NetMan::ReplaceSubCktPro(vector <ll> vLI, vector <ll> vLO, Abc_Ntk_t * pSubNtk, vector <ll> vLO_ori) {
    // cout << "ReplaceSubCktPro begin!" << endl;
    if (vLO.size() != Abc_NtkPoNum(pSubNtk)) {
        cout << "vLO.size() = " << vLO.size() << ", Abc_NtkPoNum(pSubNtk) = " << Abc_NtkPoNum(pSubNtk) << endl;
        assert(0);
    }
    if (vLI.size() != Abc_NtkPiNum(pSubNtk)) {
        cout << "vLI.size() = " << vLI.size() << ", Abc_NtkPiNum(pSubNtk) = " << Abc_NtkPiNum(pSubNtk) << endl;
        assert(0);
    }
    Abc_Obj_t * pNode, * pFanout, * pFanin;
    ll i, j;
    // duplicate the nodes
    Abc_NtkCleanCopy(pSubNtk);
    Abc_NtkForEachNodeReverse(pSubNtk, pNode, i) {
        Abc_NtkDupObj(pNtk, pNode, 0);  // copy node to current pNtk
    }

    // Duplicated subnetwork nodes in host network: never patch LO fanins away from LI
    // when LI and LO share the same host net (those edges must stay on the boundary node).
    std::unordered_set<Abc_Obj_t *> subDupFanouts;
    Abc_NtkForEachNode(pSubNtk, pNode, i)
        subDupFanouts.insert(pNode->pCopy);

    // connect all objects
    // cout << "connect all objects begin!" << endl;
    ll firstPoId = Abc_ObjId(Abc_NtkPo(pSubNtk, 0));
    Abc_NtkForEachNodeReverse(pSubNtk, pNode, i) {
        if (Abc_ObjIsPo(Abc_ObjFanout0(pNode))) {     // PO driver
            Abc_Obj_t * pPo = Abc_ObjFanout0(pNode);
            ll iPo = pPo->Id - firstPoId;
            
            assert(iPo < vLO.size());
            assert(vLO[iPo] < GetIdMaxPlus1());
            Abc_Obj_t * pLO = GetObj(vLO[iPo]);
            Vec_Ptr_t * vObjFanouts = Vec_PtrAlloc(Abc_ObjFanoutNum(pLO));
            Abc_NodeCollectFanouts(pLO, vObjFanouts);
            Vec_PtrForEachEntry(Abc_Obj_t *, vObjFanouts, pFanout, j) {
                if (subDupFanouts.count(pFanout))
                    continue;
                Abc_ObjPatchFanin(pFanout, pLO, pNode->pCopy);
            }
            Vec_PtrFree(vObjFanouts);

            // update newIdMap
            assert(GetNewId(vLO_ori[iPo]) == pLO->Id);
            newIdMap[vLO_ori[iPo]] = pNode->pCopy->Id;
            pNode->pCopy->oriId = vLO_ori[iPo];
            // if (vLO_ori[iPo] == 12176) {
            //     cout << "vLO_ori[iPo] == 12176:";
            //     cout << " pLO->Id = " << pLO->Id << ", pNode->pCopy->Id = " << pNode->pCopy->Id << endl;
            // }
        }
        
        Abc_ObjForEachFanin(pNode, pFanin, j) {
            if (!Abc_ObjIsPi(pFanin)) {
                assert(!Abc_ObjIsPo(pFanin->pCopy));
                assert(!Abc_ObjIsPi(pNode->pCopy));
                Abc_ObjAddFanin(pNode->pCopy, pFanin->pCopy);
            }
            else {
                for (int k = 0; k < vLI.size(); k++) {
                    assert(k < Abc_NtkPiNum(pSubNtk));
                    if (pFanin == Abc_NtkPi(pSubNtk, k)) {
                        assert(!Abc_ObjIsPo(Abc_NtkObj(pNtk, vLI[k])));
                        assert(!Abc_ObjIsPi(pNode->pCopy));
                        Abc_ObjAddFanin(pNode->pCopy, Abc_NtkObj(pNtk, vLI[k]));
                    }
                }
            }
        }
    }

    // PO driven only by PI: no internal node iteration above; still rewire external LO fanouts.
    for (int ii = 0; ii < Abc_NtkPoNum(pSubNtk); ++ii) {
        Abc_Obj_t * pPo = Abc_NtkPo(pSubNtk, ii);
        Abc_Obj_t * pDrv = Abc_ObjFanin0(pPo);
        if (!Abc_ObjIsPi(pDrv))
            continue;
        int kPi = -1;
        for (int kk = 0; kk < Abc_NtkPiNum(pSubNtk); ++kk) {
            if (Abc_NtkPi(pSubNtk, kk) == pDrv) {
                kPi = kk;
                break;
            }
        }
        assert(kPi >= 0);
        ll iPo = pPo->Id - firstPoId;
        assert(iPo >= 0 && iPo < (ll)vLO.size());
        Abc_Obj_t * pLO = GetObj(vLO[iPo]);
        Abc_Obj_t * pNewDrv = Abc_NtkObj(pNtk, vLI[kPi]);
        if (pNewDrv != pLO) {
            Vec_Ptr_t * vObjFanouts = Vec_PtrAlloc(Abc_ObjFanoutNum(pLO));
            Abc_NodeCollectFanouts(pLO, vObjFanouts);
            Vec_PtrForEachEntry(Abc_Obj_t *, vObjFanouts, pFanout, j) {
                if (subDupFanouts.count(pFanout))
                    continue;
                Abc_ObjPatchFanin(pFanout, pLO, pNewDrv);
            }
            Vec_PtrFree(vObjFanouts);
        }
        assert(GetNewId(vLO_ori[iPo]) == pLO->Id);
        newIdMap[vLO_ori[iPo]] = pNewDrv->Id;
        pNewDrv->oriId = vLO_ori[iPo];
    }

    // delete original sub-circuit
    // cout << "delete original sub-circuit begin!" << endl;

    // avoid deleting LIs (add fake POs)
    // for (ll i = 0; i < vLI.size(); i++) {
    //     Abc_Obj_t * pLI = GetObj(vLI[i]);
    //     if (Abc_ObjIsPo(pLI)) {
    //         cout << "pLI = " << Abc_ObjName(pLI) << " is a PO" << endl;
    //         cout << "id = " << pLI->Id << "(" << vLI[i] << ")" << endl;
    //         cout << "pLI's oriId = " << pLI->oriId << endl;
    //         assert(0);
    //     }
    //     Abc_Obj_t * pFakePO = Abc_NtkCreatePo(pNtk);
    //     assert(!Abc_ObjIsPi(pFakePO));
    //     Abc_ObjAddFanin(pFakePO, pLI); 
    // }

    std::unordered_set<ll> liNetIds(vLI.begin(), vLI.end());
    for (ll o = vLO.size() - 1; o >= 0; --o) {
        if (liNetIds.count(vLO[o]))
            continue; // LI/LO share this host node; it still feeds inserted dup nodes
        Abc_Obj_t * pLO = GetObj(vLO[o]);
        if (Abc_ObjFanoutNum(pLO) == 0) {
            // Abc_NtkDeleteObj_rec(pLO, 1);
            // cout << "delete pLO: " << Abc_ObjName(pLO) << "(id = " << pLO->Id << ", oriId = " << pLO->oriId << ")" << endl;
            Abc_NtkDeleteObj(pLO);
        }
        else {
            cout << "#fanout = " << Abc_ObjFanoutNum(pLO) << endl;
            cout << "LO(n" << vLO[o] << ")'s fanout: " << endl;
            Abc_ObjForEachFanout(pLO, pFanout, i) {
                cout << pFanout->Id << " ";
            }
            cout << endl;
            PrintPro(1, 1, 0);
            assert(0);
        }
    }
}

void NetMan::UpdateNewIdMap() {
    Abc_Obj_t * pObj;
    ll i;
    newIdMap.clear();
    bool fAll0 = true;
    Abc_NtkForEachObj(pNtk, pObj, i) {
        ll oriId = pObj->oriId;
        ll newId = pObj->Id;
        if (oriId != 0)
            fAll0 = false;
        if (oriId != newId) {
            newIdMap[oriId] = newId;
        }
    }
    assert(!fAll0);
}

bool NetMan::IsOriIdAll0(std::string mark) {
    // cout << "check IsOriIdAll0: " << mark << endl;
    Abc_Obj_t * pObj;
    ll i;
    Abc_NtkForEachObj(pNtk, pObj, i) {
        if (pObj->oriId != 0) {
            // cout << "passed" << endl;
            return false;
        }
    }
    assert(0);
    return true;
}