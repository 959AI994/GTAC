#include "error.h"


using namespace std;
using namespace abc;
using namespace boost;


ErrManPro::ErrManPro(NetMan& _net0, NetMan& _net1, bool is_sign, unsigned _seed, ll n_frame, METR_TYPE metr_type, DISTR_TYPE distr_type):
    net0(_net0), net1(_net1), pProc(nullptr), pMit(nullptr), pMitSmlt(nullptr), isSign(is_sign), seed(_seed), nFrame(n_frame), metrType(metr_type), distrType(distr_type)  {
    assert(IsPIOSame(net0, net1));
    nNetPI = net0.GetPiNum();
    nNetPo = net0.GetPoNum();
}


static void AppendNet(Abc_Ntk_t* pMitNtk, Abc_Ntk_t* pNtk0, Abc_Ntk_t* pNtk1, Abc_Ntk_t* pProcNtk, int netMark) {
    // init
    // netMark = 0, net0; netMark = 1, net1; netMark = 2, procNet;
    Abc_Ntk_t* pSubNtk = nullptr;
    if (netMark == 0)
        pSubNtk = pNtk0;
    else if (netMark == 1)
        pSubNtk = pNtk1;
    else if (netMark == 2)
        pSubNtk = pProcNtk;
    else
        assert(0);
    assert(!Abc_NtkIsStrash(pSubNtk));
    Abc_Obj_t* pObj = nullptr;
    Abc_Obj_t* pFanin = nullptr;
    int i = 0, k = 0;
    Abc_NtkCleanCopy(pSubNtk);

    // deal with PIs
    if (netMark == 0) {
        Abc_NtkForEachPi(pSubNtk, pObj, i)
            Abc_NtkDupObj(pMitNtk, pObj, 1);
    }
    else if (netMark == 1) {
        Abc_NtkForEachPi(pSubNtk, pObj, i)
            pObj->pCopy = Abc_NtkPi(pMitNtk, i);
    }
    else if (netMark == 2) {
        Abc_NtkForEachPo(pNtk0, pObj, i) {
            Abc_NtkPi(pSubNtk, i)->pCopy = Abc_ObjChild0Copy(pObj);
        }
        int nWidth = Abc_NtkPoNum(pNtk0);
        Abc_NtkForEachPo(pNtk1, pObj, i) {
            Abc_NtkPi(pSubNtk, i + nWidth)->pCopy = Abc_ObjChild0Copy(pObj);
        }
    }
    else
        assert(0);
    // duplicate nodes
    Abc_NtkForEachNode(pSubNtk, pObj, i) {
        if (pObj->pCopy == nullptr) {
            Abc_NtkDupObj(pMitNtk, pObj, 0);
            if (netMark == 0)
                RenameAbcObj(pObj->pCopy, string(Abc_ObjName(pObj)) + "_net0");
            else if (netMark == 1)
                RenameAbcObj(pObj->pCopy, string(Abc_ObjName(pObj)) + "_net1");
            else if (netMark == 2)
                RenameAbcObj(pObj->pCopy, string(Abc_ObjName(pObj)) + "_proc");
            else
                assert(0);
        }
    }
    // reconnect all nodes
    Abc_NtkForEachNode(pSubNtk, pObj, i) {
        Abc_ObjForEachFanin(pObj, pFanin, k)
            Abc_ObjAddFanin(pObj->pCopy, pFanin->pCopy);
    }
    // deal with POs
    if (netMark == 0 || netMark == 1) {
    }
    else if (netMark == 2) {
        Abc_NtkForEachPo(pSubNtk, pObj, i) {
            Abc_NtkDupObj(pMitNtk, pObj, 1);
        }
        Abc_NtkForEachPo(pSubNtk, pObj, i) {
            Abc_ObjAddFanin(pObj->pCopy, Abc_ObjChild0Copy(pObj));
        }
    }
    else
        assert(0);
}


void ErrManPro::InitMit() {
    assert(net0.GetNetType() == net1.GetNetType());
    auto netType = net0.GetNetType();

    // check/create folder
    const string folder = "designs/proc/";
    CreatePath(folder);

    // get the name of the miter file
    assert(metrType == METR_TYPE::MSE || metrType == METR_TYPE::MED);
    ostringstream fileNameBase;
    fileNameBase << folder << (isSign? "signed_": "unsigned_") << metrType << "_width_" << nNetPo;
    auto behName = fileNameBase.str() + "_beh.v";
    auto interName = fileNameBase.str() + "_inter.blif";
    auto finalName = string("");
    if (netType == NET_TYPE::GATE)
        finalName = fileNameBase.str() + "_gate.blif";
    else if (netType == NET_TYPE::SOP)
        finalName = fileNameBase.str() + "_sop.blif";
    else
        assert(0);

    // if miter file exists, load the miter file
    if (IsPathExist(finalName)) {
        AbcMan abcMan;
        abcMan.ReadNet(finalName);
        // abcMan.PrintStat();
        pProc = make_shared<NetMan>(abcMan.GetNet(), true);
    }
    else { // if miter file doesn't exist, synthesize a new miter file
        // create behavior-level Verilog
        CreateBehLevMit(behName);
        // call yosys for HLS
        ostringstream comm;
        comm << "yosys -q -p \"read_verilog " << behName << "; synth; write_blif " << interName << "\"";
        ExecSystComm(comm.str());
        // further synthesis
        AbcMan abcMan;
        abcMan.ReadNet(interName);
        abcMan.Comm("st; ps; resyn2; ps; resyn2; ps; resyn2; ps;");
        if (netType == NET_TYPE::GATE) {
            abcMan.Comm("dch; amap; ps;");
            abcMan.WriteNet(finalName);
        }
        else if (netType == NET_TYPE::SOP) {
            abcMan.Comm("if -K 6 -a; ps;");
            abcMan.WriteNet(finalName);
        }
        else
            assert(0);
        abcMan.ReadNet(finalName);
        pProc = make_shared<NetMan>(abcMan.GetNet(), true);
    }

    // build miter
    // init
    pMit = make_shared<NetMan>();
    pMit->StartSopNet();
    // copy net0
    // net0.WriteNet("tmp/net0.blif");
    AppendNet(pMit->GetNet(), net0.GetNet(), net1.GetNet(), pProc->GetNet(), 0);
    // net1.WriteNet("tmp/net1.blif");
    AppendNet(pMit->GetNet(), net0.GetNet(), net1.GetNet(), pProc->GetNet(), 1);
    AppendNet(pMit->GetNet(), net0.GetNet(), net1.GetNet(), pProc->GetNet(), 2);
    // pMit->PrintStat();
}


void ErrManPro::CreateBehLevMit(const string& fileName) {
    FILE* f = fopen(fileName.c_str(), "w");
    fprintf(f, "module proc(a, b, f);\n");
    fprintf(f, "parameter width = %d;\n", nNetPo);
    if (isSign) {
        fprintf(f, "input signed [width - 1: 0] a;\n");
        fprintf(f, "input signed [width - 1: 0] b;\n");
    }
    else {
        fprintf(f, "input [width - 1: 0] a;\n");
        fprintf(f, "input [width - 1: 0] b;\n");
    }
    if (metrType == METR_TYPE::MSE)
        fprintf(f, "output [width * 2 - 1: 0] f;\n");
    else if (metrType == METR_TYPE::MED)
        fprintf(f, "output [width - 1: 0] f;\n");
    else
        assert(0);
    fprintf(f, "wire [width - 1: 0] diff;\n");
    fprintf(f, "assign diff = (a > b)? (a - b): (b - a);\n");
    if (metrType == METR_TYPE::MSE)
        fprintf(f, "assign f = diff * diff;\n");
    else if (metrType == METR_TYPE::MED)
        fprintf(f, "assign f = diff;\n");
    else
        assert(0);
    fprintf(f, "endmodule\n");
    fclose(f);
}


double ErrManPro::CalcErr() {
    pMitSmlt = make_shared <Simulator> (*pMit, seed, nFrame);
    if (distrType == DISTR_TYPE::UNIF) {
        pMitSmlt->InpUnifFast();
    }
    else if (distrType == DISTR_TYPE::ENUM) {
        pMitSmlt->InpEnum();
    }
    else if (distrType == DISTR_TYPE::MIX) {
        pMitSmlt->InpMix();
    }
    else
        assert(0);
    pMitSmlt->Sim();
    bigInt res = 0;
    for (int i = pMitSmlt->GetPoNum() - 1; i >= 0; --i) {
        res <<= 1;
        res += pMitSmlt->CountNumbOfOnes(pMitSmlt->GetPoId(i));
    }
    auto ret = static_cast<double>(static_cast<bigFlt>(res) / static_cast<bigFlt>(nFrame));
    return ret;
}


ErrMan::ErrMan(NetMan & netMan0, NetMan & netMan1, unsigned _seed, ll n_frame, ll n_output, DISTR_TYPE distr_type):
    net0(netMan0), net1(netMan1), pSmlt0(nullptr), pSmlt1(nullptr), seed(_seed), nFrame(n_frame), nOutput(n_output), distrType(distr_type) {
    #ifdef DEBUG
    assert(IsPIOSame(net0, net1));
    #endif
}


void ErrMan::InitForStatErr() {
    if (pSmlt0 != nullptr || pSmlt1 != nullptr) {
        assert(pSmlt0 != nullptr && pSmlt1 != nullptr);
        return;
    }
    pSmlt0 = make_shared <Simulator> (net0, seed, nFrame);
    pSmlt1 = make_shared <Simulator> (net1, seed, nFrame); 
    if (distrType == DISTR_TYPE::UNIF) {
        pSmlt0->InpUnifFast();
        pSmlt1->InpUnifFast();
    }
    else if (distrType == DISTR_TYPE::ENUM) {
        pSmlt0->InpEnum();
        pSmlt1->InpEnum();
    }
    else if (distrType == DISTR_TYPE::MIX) {
        pSmlt0->InpMix();
        pSmlt1->InpMix();
    }
    // else if (distrType == DISTR_TYPE::SELF) {
    //     pSmlt0->InpSelf(selfDefDistr);
    //     pSmlt1->InpSelf(selfDefDistr);
    // }
    else
        assert(0);
    pSmlt0->Sim();
    pSmlt1->Sim();

    auto topoNodes = net1.TopoSort();
    for (const auto & pNode: topoNodes) {
        int num1 = pSmlt1->CountNumbOfOnes(pNode->Id);
        pNode->numof1s = num1;
        pNode->numof0s = nFrame - num1;
    }
}


double ErrMan::CalcErrRate(bool isSign, ll nOutput, vector <ll> RealCom, ll cutId) {
    InitForStatErr();
    return pSmlt0->GetErrRate(*pSmlt1, isSign, nOutput, RealCom, cutId);
}


double ErrMan::CalcMeanErrDist(bool isSign, ll nOutput, vector <ll> RealCom, ll cutId) {
    InitForStatErr();
    return pSmlt0->GetMeanErrDist(*pSmlt1, isSign, nOutput, RealCom, cutId);
}


double ErrMan::CalcMeanErr(bool isSign, ll nOutput, vector <ll> RealCom, ll cutId) {
    InitForStatErr();
    return pSmlt0->GetMeanErr(*pSmlt1, isSign, nOutput, RealCom, cutId);
}


double ErrMan::CalcMeanSquareErr(bool isSign, ll nOutput, vector <ll> RealCom, ll cutId) {
    InitForStatErr();
    return pSmlt0->GetMeanSquareErr(*pSmlt1, isSign, nOutput, RealCom, cutId);
}

double ErrMan::CalcMeanSquareErr_forDebug(bool isSign, ll nOutput, vector <ll> RealCom, ll cutId) {
    InitForStatErr();
    return pSmlt0->GetMeanSquareErr_forDebug(*pSmlt1, isSign, nOutput, RealCom, cutId);
}


double ErrMan::CalcSigNoiseRat(bool isSign, ll nOutput, vector <ll> RealCom, ll cutId) {
    InitForStatErr();
    return pSmlt0->GetSigNoiseRat(*pSmlt1, isSign, nOutput, RealCom, cutId);
}

double ErrMan::CalcMaxErrDist(bool isSign, ll nOutput, vector <ll> RealCom, ll cutId) {
    InitForStatErr();
    return pSmlt0->GetMaxErrDist(*pSmlt1, isSign, nOutput, RealCom, cutId);
}

double ErrMan::CalcMeanRelErrDist(bool isSign, ll nOutput, vector <ll> RealCom, ll cutId) {
    InitForStatErr();
    return pSmlt0->GetMeanRelErrDist(*pSmlt1, isSign, nOutput, RealCom, cutId);
}

double ErrMan::CalcMeanHamDist(bool isSign, ll nOutput, vector <ll> RealCom, ll cutId) {
    InitForStatErr();
    return pSmlt0->GetMeanHamDist(*pSmlt1, isSign, nOutput, RealCom, cutId);
}


// double ErrMan::CalcSelfDefErr(bool isSign, const string & selfDefMetr) {
//     InitForStatErr();
//     return pSmlt0->GetSelfDefErr(*pSmlt1, isSign, selfDefMetr);
// }


// ull ErrMan::CalcMaxErrDist(bool isSign) {
//     #ifdef DEBUG
//     assert(net0.GetPiNum() <= 60 && net1.GetPiNum() <= 60);
//     assert(isSign == 0);
//     #endif
//     auto pNtk0 = Abc_NtkDup(net0.GetNet());
//     auto pNtk1 = Abc_NtkDup(net1.GetNet());
//     ull res = GET_MEM(pNtk0, pNtk1);
//     Abc_NtkDelete(pNtk0);
//     Abc_NtkDelete(pNtk1);
//     return res;
// }


ull ErrMan::GET_MEM(Abc_Ntk_t * pNtk1, Abc_Ntk_t * pNtk2) {
    ull pTemp = 0;
    ll fRemove1, fRemove2;
    assert( Abc_NtkHasOnlyLatchBoxes(pNtk1) );
    assert( Abc_NtkHasOnlyLatchBoxes(pNtk2) );
    fRemove1 = (!Abc_NtkIsStrash(pNtk1) || Abc_NtkGetChoiceNum(pNtk1)) && (pNtk1 = Abc_NtkStrash(pNtk1, 0, 0, 0));
    fRemove2 = (!Abc_NtkIsStrash(pNtk2) || Abc_NtkGetChoiceNum(pNtk2)) && (pNtk2 = Abc_NtkStrash(pNtk2, 0, 0, 0));
    if ( pNtk1 && pNtk2 )
        pTemp = NtkMiterComp( pNtk1, pNtk2 );
    if ( fRemove1 )  Abc_NtkDelete( pNtk1 );
    if ( fRemove2 )  Abc_NtkDelete( pNtk2 );
    return pTemp;
}


static void Abc_NtkMiterPrepare( Abc_Ntk_t * pNtk1, Abc_Ntk_t * pNtk2, Abc_Ntk_t * pNtkMiter) {
    Abc_Obj_t * pObj, * pObjNew; ll i;
    Abc_AigConst1(pNtk1)->pCopy = Abc_AigConst1(pNtkMiter);
    Abc_AigConst1(pNtk2)->pCopy = Abc_AigConst1(pNtkMiter);

    // create new PIs and remember them in the old PIs
    Abc_NtkForEachPi(pNtk1, pObj, i)
    {
        pObjNew = Abc_NtkCreatePi(pNtkMiter);
        // remember this PI in the old PIs
        pObj->pCopy = pObjNew;
        pObj = Abc_NtkPi(pNtk2, i);  
        pObj->pCopy = pObjNew;
            // add name
        Abc_ObjAssignName( pObjNew, Abc_ObjName(pObj), NULL );
    }

        // pObjNew = Abc_NtkCreatePo(pNtkMiter);
        // Abc_ObjAssignName(pObjNew, "miter", NULL);

    Abc_NtkForEachLatch( pNtk1, pObj, i )
    {
        pObjNew = Abc_NtkDupBox( pNtkMiter, pObj, 0 );
        // add names
        Abc_ObjAssignName( pObjNew, Abc_ObjName(pObj), "_1" );
        Abc_ObjAssignName( Abc_ObjFanin0(pObjNew),  Abc_ObjName(Abc_ObjFanin0(pObj)), "_1" );
        Abc_ObjAssignName( Abc_ObjFanout0(pObjNew), Abc_ObjName(Abc_ObjFanout0(pObj)), "_1" );
    }
    Abc_NtkForEachLatch( pNtk2, pObj, i )
    {
        pObjNew = Abc_NtkDupBox( pNtkMiter, pObj, 0 );
        // add name
        Abc_ObjAssignName( pObjNew, Abc_ObjName(pObj), "_2" );
        Abc_ObjAssignName( Abc_ObjFanin0(pObjNew),  Abc_ObjName(Abc_ObjFanin0(pObj)), "_2" );
        Abc_ObjAssignName( Abc_ObjFanout0(pObjNew), Abc_ObjName(Abc_ObjFanout0(pObj)), "_2" );
    }
}


static void Abc_NtkMiterAddOne( Abc_Ntk_t * pNtk, Abc_Ntk_t * pNtkMiter ) {
    Abc_Obj_t * pNode;
    ll i;
    assert( Abc_NtkIsDfsOrdered(pNtk) );
    Abc_AigForEachAnd( pNtk, pNode, i )
        pNode->pCopy = Abc_AigAnd( (Abc_Aig_t *)pNtkMiter->pManFunc, Abc_ObjChild0Copy(pNode), Abc_ObjChild1Copy(pNode) );
}


ull ErrMan::NtkMiterComp(Abc_Ntk_t * pNtk1, Abc_Ntk_t * pNtk2) {
    const ll fImplic = 0, fComb = 0, nPartSize = 0, fMulti = 0;
    char Buffer[1000];
    Abc_Ntk_t * pNtkMiter;

    assert( Abc_NtkIsStrash(pNtk1) );
    assert( Abc_NtkIsStrash(pNtk2) );

    // start the new network
    pNtkMiter = Abc_NtkAlloc(ABC_NTK_STRASH, ABC_FUNC_AIG, 1);
    sprintf( Buffer, "%s_%s_miter", pNtk1->pName, pNtk2->pName );
    pNtkMiter->pName = Extra_UtilStrsav(Buffer);

    // perform strashing
    Abc_NtkMiterPrepare( pNtk1, pNtk2, pNtkMiter );
    Abc_NtkMiterAddOne( pNtk1, pNtkMiter ); 
    Abc_NtkMiterAddOne( pNtk2, pNtkMiter );
    ull x = NtkMiterFinalize( pNtk1, pNtk2, pNtkMiter, fComb, nPartSize, fImplic, fMulti );
    
    Abc_AigCleanup((Abc_Aig_t *)pNtkMiter->pManFunc);

    // make sure that everything is okay
    Abc_NtkDelete( pNtkMiter );
    return x;
}


ull ErrMan::NtkMiterFinalize( Abc_Ntk_t * pNtk1, Abc_Ntk_t * pNtk2, Abc_Ntk_t * pNtkMiter, ll fComb, ll nPartSize, ll fImplic, ll fMulti ) {
    Vec_Ptr_t * vPairs;
    Abc_Obj_t * pNode;
    ll i;
    assert( nPartSize == 0 || fMulti == 0 );
    // collect the PO pairs from both networks
    vPairs = Vec_PtrAlloc( 100 );
    // collect the PO nodes for the miter
    Abc_NtkForEachPo( pNtk1, pNode, i )
    {
        Vec_PtrPush( vPairs, Abc_ObjChild0Copy(pNode) );
        pNode = Abc_NtkPo( pNtk2, i );
        Vec_PtrPush( vPairs, Abc_ObjChild0Copy(pNode) );
    }
    Abc_NtkForEachLatch( pNtk1, pNode, i )
        Abc_ObjAddFanin( Abc_ObjFanin0(pNode)->pCopy, Abc_ObjChild0Copy(Abc_ObjFanin0(pNode)) );
    Abc_NtkForEachLatch( pNtk2, pNode, i )
        Abc_ObjAddFanin( Abc_ObjFanin0(pNode)->pCopy, Abc_ObjChild0Copy(Abc_ObjFanin0(pNode)) );
    
    // add the miter
    // Abc_Obj_t * NewPO = Abc_NtkPo(pNtkMiter, 0);
    // Abc_ObjAddFanin( NewPO, pMiter );
    
    Abc_Obj_t ** X = new Abc_Obj_t *[vPairs->nSize / 2];
    Abc_Obj_t ** Y = new Abc_Obj_t *[vPairs->nSize / 2];
    for(i = 0; i < vPairs->nSize; i += 2)
    {
        X[i / 2] = (Abc_Obj_t *) vPairs->pArray[i];
        Y[i / 2] = (Abc_Obj_t *) vPairs->pArray[i + 1];
    }
            // for(i=0; i<vPairs->nSize; i += 2)
            // {
            //     Abc_Obj_t * NewPo = Abc_NtkCreatePo(pNtkMiter);
            //     Abc_ObjAddFanin(NewPo, (Abc_Obj_t *)vPairs->pArray[i]);
            // }
            // for(i=0; i<vPairs->nSize; i += 2)
            // {
            //     Abc_Obj_t * NewPo = Abc_NtkCreatePo(pNtkMiter);
            //     Abc_ObjAddFanin(NewPo, (Abc_Obj_t *)vPairs->pArray[i+1]);
            // }
            // Ckt_WriteBlif(pNtkMiter, "first.blif"); 
    Abc_Obj_t ** R = X_subtract_Y_abs(pNtkMiter, X, Y, vPairs->nSize / 2);     
    delete[] X;
    delete[] Y;
    ull res = GETMEM(pNtkMiter, R, vPairs->nSize / 2);
    delete[] R;
            // cout << res << endl;

    Vec_PtrFree( vPairs );
    return res;
}


ull ErrMan::GETMEM(Abc_Ntk_t * pNtk, Abc_Obj_t *R[], ll n) {
    Abc_Obj_t * ConstNode[2];
    Abc_Obj_t ** mem = new Abc_Obj_t *[n];
    ConstNode[1] = Abc_AigConst1(pNtk);
    ConstNode[0] = Abc_ObjNot(ConstNode[1]);
    bool CurrentState = true, PreviousState = true;
    ll round = 0;
    std::vector<ll> ConstMEMInput(n, 0);
    ConstMEMInput[n - 1] = 1;
    Abc_Obj_t * tempR;
    while(true)
    {
        // for(ll k = 0; k < n; k++) cout << ConstMEMInput[n-1-k];
        // cout << round << endl;
        Abc_Obj_t * result = Abc_NtkCreatePo(pNtk);
        for(ll k = 0; k < n; k++) mem[k] = ConstNode[ConstMEMInput[k]];
        // compXY = (R > mem)
        tempR = X_lt_Y(pNtk, mem, R, n);
        Abc_ObjAddFanin(result, tempR);
        // Ckt_WriteBlif(pNtk, "merge_SAT.blif");
        PreviousState = CurrentState;
        CurrentState = SATSolver(pNtk);
        if(round == n) break;
        if(CurrentState)
        {
            if(round < n - 1) ConstMEMInput[n - 1 - ++round] = 1;
            else break;
        }
        else 
        {
            if (round < n-1) {ConstMEMInput[n - 1 - round++] = 0; ConstMEMInput[n - 1 - round] = 1;} 
            else ConstMEMInput[n - 1 - round++] = 0;
        }
        Abc_NtkDeleteObj(result);
    }
    delete[] mem; 
    ull res = 0;
    for(ll k = 0; k < n; k++) {
        res <<= 1;
        res += ConstMEMInput[n-1-k];
        // cout << res << endl;
    }
    if(CurrentState) res++; 
    return res;
}


Abc_Obj_t ** ErrMan::X_subtract_Y_abs(Abc_Ntk_t * pNtk, Abc_Obj_t * X[], Abc_Obj_t * Y[], ll n) {
    if(n <= 0) return nullptr;
    Abc_Obj_t ** R = new Abc_Obj_t *[n];
    Abc_Obj_t ** Cout = new Abc_Obj_t *[n];
    R[0] = Abc_AigXor((Abc_Aig_t *) pNtk->pManFunc, X[0], Y[0]);
    Cout[0] = Abc_AigAnd((Abc_Aig_t *) pNtk->pManFunc, Abc_ObjNot(X[0]), Y[0]);
    for(ll i=1; i<n; i++)
    {
        R[i] = Abc_AigXor((Abc_Aig_t *) pNtk->pManFunc, 
            Abc_AigXor((Abc_Aig_t *) pNtk->pManFunc, Cout[i-1], X[i]), 
            Y[i]
        );
        Abc_Obj_t * temp1 = Abc_AigAnd((Abc_Aig_t *) pNtk->pManFunc, Abc_ObjNot(X[i]), Y[i]);
        Abc_Obj_t * temp2 = Abc_AigOr((Abc_Aig_t *) pNtk->pManFunc, Abc_ObjNot(X[i]), Y[i]);
        Cout[i] = Abc_AigOr((Abc_Aig_t *) pNtk->pManFunc,
            Abc_AigAnd((Abc_Aig_t *) pNtk->pManFunc, temp1, Abc_ObjNot(Cout[i-1])),
            Abc_AigAnd((Abc_Aig_t *) pNtk->pManFunc, temp2, Cout[i-1])
        );
    }

    // 2's complement of R ( = R' + 1 )
    Abc_Obj_t ** R_2Complement = new Abc_Obj_t *[n];
    Abc_Obj_t ** Cout_2 = new Abc_Obj_t * [n];
    R_2Complement[0] = R[0];
    Cout_2[0] = Abc_ObjNot(R[0]);
    for(ll k=1; k<n; k++)
    {
        R_2Complement[k] = Abc_AigXor((Abc_Aig_t *) pNtk->pManFunc, Cout_2[k-1], Abc_ObjNot(R[k]));
        Cout_2[k] = Abc_AigAnd((Abc_Aig_t *) pNtk->pManFunc, Cout_2[k-1], Abc_ObjNot(R[k]));
    }
    Abc_Obj_t ** res = new Abc_Obj_t *[n];
    for(ll k=0; k<n; k++)
        res[k] = Abc_AigMux((Abc_Aig_t *) pNtk->pManFunc, Cout[n-1], R_2Complement[k], R[k]);
    delete[] R_2Complement;
    delete[] Cout;
    delete[] Cout_2;
    delete[] R;
    return res;
}


Abc_Obj_t * ErrMan::X_lt_Y(Abc_Ntk_t * pNtk, Abc_Obj_t * X[], Abc_Obj_t * Y[], ll n) {
    /* implementation */
    if(n <= 0) return nullptr;
    std::vector<Abc_Obj_t *> PreBitsComp(n, nullptr);
    PreBitsComp[0] = Abc_AigAnd((Abc_Aig_t *) pNtk->pManFunc, Abc_ObjNot(X[0]), Y[0]);
    for(ll i=1; i<n; i++)
    {
        PreBitsComp[i] = 
            Abc_AigOr((Abc_Aig_t *) pNtk->pManFunc, 
                Abc_AigAnd((Abc_Aig_t *) pNtk->pManFunc, Abc_ObjNot(X[i]), Y[i]),
                Abc_AigAnd((Abc_Aig_t *) pNtk->pManFunc, PreBitsComp[i-1],  
                    Abc_ObjNot(Abc_AigXor((Abc_Aig_t *) pNtk->pManFunc, X[i], Y[i]))
                )
            );
    }
    return PreBitsComp[n - 1];
}


bool ErrMan::SATSolver(Abc_Ntk_t * pNtk) {
    ll RetValue = -1;
    ll fVerbose = 0;
    ll nConfLimit = 0;
    ll nInsLimit = 0;
    assert(pNtk != nullptr);
    assert( Abc_NtkIsStrash(pNtk) );
    RetValue = Abc_NtkMiterSat( pNtk, (ABC_INT64_T)nConfLimit, (ABC_INT64_T)nInsLimit, fVerbose, NULL, NULL );
    if (pNtk->pModel != nullptr)
        ABC_FREE(pNtk->pModel);
    if (RetValue == -1)
        assert(0);
    else if (RetValue == 0)
        return true;
    else
        return false;
    return 0;
}


double CalcErrPro(NetMan& net0, NetMan& net1, bool isSign, unsigned seed, ll nFrame, METR_TYPE metrType, DISTR_TYPE distrType) {
    ErrManPro errMan(net0, net1, isSign, seed, nFrame, metrType, distrType);
    errMan.InitMit();
    auto err = errMan.CalcErr();
    return err;
}


double CalcErr(NetMan & netMan0, NetMan & netMan1, bool isSign, unsigned seed, ll nFrame, ll nOutput, METR_TYPE metrType, DISTR_TYPE distrType, vector <ll> RealCom, ll cutId) {
    ErrMan errMan(netMan0, netMan1, seed, nFrame, nOutput, distrType);
    if (metrType == METR_TYPE::ER)
        return errMan.CalcErrRate(isSign, nOutput, RealCom, cutId);
    else if (metrType == METR_TYPE::MED)
        return errMan.CalcMeanErrDist(isSign, nOutput, RealCom, cutId);
    else if (metrType == METR_TYPE::ME)
        return errMan.CalcMeanErr(isSign, nOutput, RealCom, cutId);
    else if (metrType == METR_TYPE::MSE)
        return errMan.CalcMeanSquareErr(isSign, nOutput, RealCom, cutId);
    else if (metrType == METR_TYPE::SNR)
        return errMan.CalcSigNoiseRat(isSign, nOutput, RealCom, cutId);
    else if (metrType == METR_TYPE::MAXED)
        return errMan.CalcMaxErrDist(isSign, nOutput, RealCom, cutId);
    // else if (metrType == METR_TYPE::SELF)
    //     return errMan.CalcSelfDefErr(isSign, selfDefMetr);
    else if (metrType == METR_TYPE::MRED)
        return errMan.CalcMeanRelErrDist(isSign, nOutput, RealCom, cutId);
    else if (metrType == METR_TYPE::MHD)
        return errMan.CalcMeanHamDist(isSign, nOutput, RealCom, cutId);
    else {
        assert(0);
        return 0;
    }
}

double CalcErr_forDebug(NetMan & netMan0, NetMan & netMan1, bool isSign, unsigned seed, ll nFrame, ll nOutput, METR_TYPE metrType, DISTR_TYPE distrType, vector <ll> RealCom, ll cutId) {
    ErrMan errMan(netMan0, netMan1, seed, nFrame, nOutput, distrType);
    if (metrType == METR_TYPE::ER)
        return errMan.CalcErrRate(isSign, nOutput, RealCom, cutId);
    else if (metrType == METR_TYPE::MED)
        return errMan.CalcMeanErrDist(isSign, nOutput, RealCom, cutId);
    else if (metrType == METR_TYPE::ME)
        return errMan.CalcMeanErr(isSign, nOutput, RealCom, cutId);
    else if (metrType == METR_TYPE::MSE)
        return errMan.CalcMeanSquareErr_forDebug(isSign, nOutput, RealCom, cutId);
    else if (metrType == METR_TYPE::SNR)
        return errMan.CalcSigNoiseRat(isSign, nOutput, RealCom, cutId);
    else if (metrType == METR_TYPE::MAXED)
        return errMan.CalcMaxErrDist(isSign, nOutput, RealCom, cutId);
    // else if (metrType == METR_TYPE::SELF)
    //     return errMan.CalcSelfDefErr(isSign, selfDefMetr);
    else if (metrType == METR_TYPE::MRED)
        return errMan.CalcMeanRelErrDist(isSign, nOutput, RealCom, cutId);
    else if (metrType == METR_TYPE::MHD)
        return errMan.CalcMeanRelErrDist(isSign, nOutput, RealCom, cutId);
    else {
        assert(0);
        return 0;
    }
}


double GetMSEFromSNR(NetMan & net, bool isSign, unsigned seed, ll nFrame, DISTR_TYPE distrType, double snr, ll nOutput) {
    Simulator smlt(net, seed, nFrame);
    if (distrType == DISTR_TYPE::ENUM)
        smlt.InpEnum();
    else if (distrType == DISTR_TYPE::UNIF)
        smlt.InpUnifFast();
    else if (distrType == DISTR_TYPE::MIX)
        smlt.InpMix();
    else
        assert(0);
    smlt.Sim();
    bigInt sumAcc2 = 0;
    for (ll i = 0; i < nFrame; ++i) {
        auto accOut = smlt.GetOutpPro_complex(i, isSign, nOutput);
        // cout << accOut[0] << endl;
        for (ll j = 0; j < nOutput; ++j)
            sumAcc2 += accOut[j] * accOut[j];
    }
    return static_cast <double> (bigFlt(sumAcc2) / bigFlt(nFrame) / bigFlt(pow(bigFlt(10),  bigFlt(snr) / 10)));
}


void VECBEEMan::BatchErrEstPro(NetMan & accNet, NetMan & appNet, LACMan & lacMan, const bigInt & uppBound, bool useAppDisjCut, ll nOutput, vector <ll> RealCom) {
    #ifdef DEBUG
    assert(IsPIOSame(accNet, appNet));
    #endif
    auto topoNodes = appNet.TopoSort();
    if (useAppDisjCut)
        FindAppDisjCut(appNet);
    else
        FindDisjCut(appNet, topoNodes);
    Simulator accSmlt(accNet, seed, nFrame);
    Simulator appSmlt(appNet, seed, nFrame);
    if (distrType == DISTR_TYPE::UNIF) {
        accSmlt.InpUnifFast();
        appSmlt.InpUnifFast();
    }
    else if (distrType == DISTR_TYPE::ENUM) {
        accSmlt.InpEnum();
        appSmlt.InpEnum();
    }
    else if (distrType == DISTR_TYPE::MIX) {
        accSmlt.InpMix();
        appSmlt.InpMix();
    }
    else
        assert(0);
    accSmlt.Sim();
    appSmlt.Sim();
    CalcBoolDiffCut2Node(appSmlt, topoNodes);
    CalcBoolDiffPo2NodePlus(appSmlt, topoNodes);
    CalcLACErrsPlus(accSmlt, appSmlt, lacMan, uppBound, nOutput, useAppDisjCut, RealCom);
}

void VECBEEMan::BatchErrEstPro_GetLacPerNetNode(NetMan & accNet, NetMan & appNet, LACMan & lacMan, const bigInt & uppBound, bool useAppDisjCut, ll nOutput, vector <ll> RealCom, unordered_map <ll, std::shared_ptr <LAC>> & LacPerNode, bool fFilt) {
    #ifdef DEBUG
    assert(IsPIOSame(accNet, appNet));
    #endif
    auto topoNodes = appNet.TopoSort();
    if (useAppDisjCut)
        FindAppDisjCut(appNet);
    else
        FindDisjCut(appNet, topoNodes);
    Simulator accSmlt(accNet, seed, nFrame);
    Simulator appSmlt(appNet, seed, nFrame);
    cout << "batchErrEsti use seed " << seed << endl;
    if (distrType == DISTR_TYPE::UNIF) {
        accSmlt.InpUnifFast();
        appSmlt.InpUnifFast();
    }
    else if (distrType == DISTR_TYPE::ENUM) {
        accSmlt.InpEnum();
        appSmlt.InpEnum();
    }
    else if (distrType == DISTR_TYPE::MIX) {
        accSmlt.InpMix();
        appSmlt.InpMix();
    }
    else
        assert(0);
    accSmlt.Sim();
    appSmlt.Sim();

    // for (const auto & pNode: topoNodes) {
    //     int num1 = appSmlt.CountNumbOfOnes(pNode->Id);
    //     pNode->numof1s = num1;
    //     pNode->numof0s = nFrame - num1;
    // }

    CalcBoolDiffCut2Node(appSmlt, topoNodes);
    CalcBoolDiffPo2NodePlus(appSmlt, topoNodes);
    CalcLACErrsPlus_GetLacPerNode(accSmlt, appSmlt, lacMan, uppBound, nOutput, useAppDisjCut, RealCom, LacPerNode, appNet.GetIdMaxPlus1(), fFilt, appNet);
}


void VECBEEMan::FindDisjCut(NetMan & net, vector <Abc_Obj_t *> & topoNodes) {
    cout << "finding disjoint cuts" << endl;
    #ifdef DEBUG
    assert(disjCuts.empty());
    assert(cutNtks.empty());
    assert(topoNodes.size());
    assert(topoNodes[0]->pNtk == net.GetNet());
    #endif

    // init
    cutNtks.resize(net.GetIdMaxPlus1());
    disjCuts.resize(net.GetIdMaxPlus1());
    poMarks.resize(net.GetIdMaxPlus1(), dynamic_bitset <ull>(net.GetPoNum(), 0));
    for (ll i = 0; i < net.GetIdMaxPlus1(); ++i) {
        if (net.GetObj(i) == nullptr)
            continue;
        poMarks[i].reset();
    }

    // update topo ids
    topoIds.resize(net.GetIdMaxPlus1());
    for (ll i = 0; i < topoNodes.size(); ++i)
        topoIds[topoNodes[i]->Id] = i;
    ll topoId = -1;
    for (ll i = 0; i < net.GetPiNum(); ++i)
        topoIds[net.GetPiId(i)] = topoId--;
    topoId = topoNodes.size();
    for (ll i = 0; i < net.GetPoNum(); ++i)
        topoIds[net.GetPoId(i)] = topoId++;

    // determine the POs that each node will affect
    for (ll i = 0; i < net.GetPoNum(); ++i)
        poMarks[net.GetPoId(i)].set(i);
    for (auto it = topoNodes.rbegin(); it != topoNodes.rend(); ++it) {
        auto pObj = *it;
        if (pObj == nullptr)
            continue;
        ll i = net.GetId(pObj);
        for (ll j = 0; j < net.GetFanoutNum(pObj); ++j)
            poMarks[i] |= poMarks[net.GetFanoutId(pObj, j)];
    }

    // collect disjoint cuts and the corresponding cut networks
    timer::progress_display pd(net.GetIdMaxPlus1());
    for (ll i = 0; i < net.GetIdMaxPlus1(); ++i) {
        auto pObj = net.GetObj(i);
        if (!net.IsNode(pObj)) {
            ++pd;
            continue;
        }
        // cout << "finding " << pObj << endl;
        Abc_NtkIncrementTravId(net.GetNet());
        FindDisjCutOfNode(pObj, disjCuts[i]);
        for (const auto & node: topoNodes) {
            if (Abc_NodeIsTravIdCurrent(node))
                cutNtks[i].emplace_back(node);
        }
        for (ll j = 0; j < net.GetPoNum(); ++j) {
            auto pPo = net.GetPo(j);
            if (Abc_NodeIsTravIdCurrent(pPo))
                cutNtks[i].emplace_back(pPo);
        }
        ++pd;
    }
}


void VECBEEMan::FindAppDisjCut(NetMan & net) {
    cout << "finding approximate disjoint cuts" << endl;
    #ifdef DEBUG
    assert(disjCuts.empty());
    assert(cutNtks.empty());
    #endif

    // init
    cutNtks.resize(net.GetIdMaxPlus1());
    disjCuts.resize(net.GetIdMaxPlus1());

    // collect disjoint cuts and the corresponding cut networks
    for (ll iNode = 0; iNode < net.GetIdMaxPlus1(); ++iNode) {
        if (!net.IsNode(iNode))
            continue;
        if (net.IsConst(iNode))
            continue;
        for (ll iFanout = 0; iFanout < net.GetFanoutNum(iNode); ++iFanout) {
            auto pFanout = net.GetFanout(iNode, iFanout);
            disjCuts[iNode].emplace_back(pFanout);
            cutNtks[iNode].emplace_back(pFanout);
        }
    }
}


void VECBEEMan::FindDisjCutOfNode(Abc_Obj_t * pObj, list <Abc_Obj_t *> & disjCut) {
    disjCut.clear();
    ExpandCut(pObj, disjCut);
    Abc_Obj_t * pObjExpd = nullptr;
    while ((pObjExpd = ExpandWhich(disjCut)) != nullptr) {
        ExpandCut(pObjExpd, disjCut);
    }
}


void ExpandCut(Abc_Obj_t * pObj, list <Abc_Obj_t *> & disjCut) {
    abc::Abc_Obj_t * pFanout = nullptr;
    ll i = 0;
    Abc_ObjForEachFanout(pObj, pFanout, i) {
        if (!abc::Abc_NodeIsTravIdCurrent(pFanout)) {
            if (abc::Abc_ObjFanoutNum(pFanout) || abc::Abc_ObjIsPo(pFanout)) {
                abc::Abc_NodeSetTravIdCurrent(pFanout);
                disjCut.emplace_back(pFanout);
            }
        }
    } 
}


Abc_Obj_t * VECBEEMan::ExpandWhich(list <Abc_Obj_t *> & disjCut) {
    for (auto ppAbcObj1 = disjCut.begin(); ppAbcObj1 != disjCut.end(); ++ppAbcObj1) {
        auto ppAbcObj2 = ppAbcObj1;
        for (++ppAbcObj2; ppAbcObj2 != disjCut.end(); ++ppAbcObj2) {
            #ifdef DEBUG
            assert(poMarks[(*ppAbcObj1)->Id].size() == poMarks[(*ppAbcObj2)->Id].size());
            assert((*ppAbcObj1)->Id != (*ppAbcObj2)->Id);
            assert(topoIds[(*ppAbcObj1)->Id] != topoIds[(*ppAbcObj2)->Id]);
            #endif
            auto isJoint = poMarks[(*ppAbcObj1)->Id] & poMarks[(*ppAbcObj2)->Id];
            if (isJoint.any()) {
                abc::Abc_Obj_t * pRet = nullptr;
                if (topoIds[(*ppAbcObj1)->Id] < topoIds[(*ppAbcObj2)->Id]) {
                    pRet = *ppAbcObj1;
                    disjCut.erase(ppAbcObj1);
                }
                else {
                    pRet = *ppAbcObj2;
                    disjCut.erase(ppAbcObj2);
                }
                return pRet;
            }
        }
    }
    return nullptr;
}


void VECBEEMan::CalcBoolDiffCut2Node(Simulator & appSmlt, vector <Abc_Obj_t *> & topoNodes) {
    cout << "calculating boolean difference of cuts with regard to nodes" << endl;
    #ifdef DEBUG
    assert(topoNodes.size());
    assert(topoNodes[0]->pNtk == appSmlt.GetNet());
    #endif
    timer::progress_display pd(topoNodes.size());
    bdCut2Nodes.resize(appSmlt.GetIdMaxPlus1());
    for (const auto & pObj: topoNodes) {
        ll i = appSmlt.GetId(pObj);
        if (!appSmlt.IsNode(pObj) || appSmlt.IsConst(pObj)) {
            ++pd;
            continue;
        }
        appSmlt.CalcLocBoolDiff(pObj, disjCuts[i], cutNtks[i], bdCut2Nodes[i]);
        ++pd;
    }
}


void VECBEEMan::CalcBoolDiffPo2Node(Simulator & appSmlt, vector <Abc_Obj_t *> & topoNodes) {
    cout << "calculating boolean difference of POs with regard to nodes" << endl;
    #ifdef DEBUG
    assert(topoNodes.size());
    assert(topoNodes[0]->pNtk == appSmlt.GetNet());
    #endif
    ll nPo = appSmlt.GetPoNum();
    bdPo2Nodes.resize(nPo);
    // timer::progress_display pd(nPo);
    for (ll o = 0; o < nPo; ++o) {
        // init boolean difference
        auto & bdPo2Node = bdPo2Nodes[o];
        bdPo2Node.resize(appSmlt.GetIdMaxPlus1(), dynamic_bitset <ull> (nFrame, 0));
        // for each PO, update boolean difference
        for (ll n = 0; n < appSmlt.GetPoNum(); ++n) {
            auto pNodeN = appSmlt.GetPo(n);
            auto nId = appSmlt.GetId(pNodeN);
            if (n == o)
                bdPo2Node[nId].set(); 
            else
                bdPo2Node[nId].reset(); 
        }
        // for each node, update boolean difference
        for (auto it = topoNodes.rbegin(); it != topoNodes.rend(); ++it) {
            auto pNodeN = *it;
            if (!appSmlt.IsNode(pNodeN))
                continue;
            ll n = appSmlt.GetId(pNodeN);
            bdPo2Node[n].reset();
            ll i = 0;
            for (auto pCut: disjCuts[n]) {
                bdPo2Node[n] |= bdPo2Node[pCut->Id] & bdCut2Nodes[n][i];
                ++i;
            } 
        }
        // ++pd;
    }
}


std::mutex mtx;
static void CalcSomeBoolDiffPo2Node(Simulator & appSmlt, std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2Nodes, vector <Abc_Obj_t *> & topoNodes, std::vector < std::list <abc::Abc_Obj_t *> > & disjCuts, std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdCut2Nodes, timer::progress_display & pd, ll start, ll end) {
    ll nFrame = appSmlt.GetFrameNumb();
    for (ll o = start; o < end; ++o) {
        auto & bdPo2Node = bdPo2Nodes[o];
        // init boolean difference
        bdPo2Node.resize(appSmlt.GetIdMaxPlus1(), dynamic_bitset <ull> (nFrame, 0));
        // for each PO, update boolean difference
        for (ll n = 0; n < appSmlt.GetPoNum(); ++n) {
            auto pNodeN = appSmlt.GetPo(n);
            auto nId = appSmlt.GetId(pNodeN);
            if (n == o)
                bdPo2Node[nId].set(); 
            else
                bdPo2Node[nId].reset(); 
        }
        // for each node, update boolean difference
        for (auto it = topoNodes.rbegin(); it != topoNodes.rend(); ++it) {
            auto pNodeN = *it;
            if (!appSmlt.IsNode(pNodeN))
                continue;
            ll n = appSmlt.GetId(pNodeN);
            bdPo2Node[n].reset();
            ll i = 0;
            for (auto pCut: disjCuts[n]) {
                bdPo2Node[n] |= bdPo2Node[pCut->Id] & bdCut2Nodes[n][i];
                ++i;
            } 
        }
        std::unique_lock<std::mutex> lock(mtx);
        ++pd;
        lock.unlock();
    }
}


void VECBEEMan::CalcBoolDiffPo2NodePlus(Simulator & appSmlt, vector <Abc_Obj_t *> & topoNodes) {
    // cout << "calculating boolean difference of POs with regard to nodes" << endl;
    #ifdef DEBUG
    assert(topoNodes.size());
    assert(topoNodes[0]->pNtk == appSmlt.GetNet());
    #endif
    ll nPo = appSmlt.GetPoNum();
    bdPo2Nodes.resize(nPo);

    ll realThread = min(nPo, nThread);
    assert(realThread > 0);
    cout << "real thread number: " << realThread << endl;
    ll chunkSize = nPo / realThread;
    ll remainder = nPo % realThread;

    timer::progress_display pd(nPo);
    vector<thread> threads;
    ll start = 0;
    for (ll i = 0; i < realThread; ++i) {
        ll end = start + chunkSize + (i < remainder? 1: 0);
        threads.emplace_back(CalcSomeBoolDiffPo2Node, std::ref(appSmlt), std::ref(bdPo2Nodes), std::ref(topoNodes), std::ref(disjCuts), std::ref(bdCut2Nodes), std::ref(pd), start, end);
        start = end;
    }
    for (auto& thread: threads)
        thread.join();
}


static void GetNewValue(Simulator & smlt, const std::vector <ll> & faninIds, const std::string & sop, boost::dynamic_bitset <ull> & value) {
    if (sop == " 0\n") {
        value.reset();
        return;
    }
    if (sop == " 1\n") {
        value.set();
        return;
    }
    char * pSop = const_cast <char *> (sop.c_str());
    ll nVars = Abc_SopGetVarNum(pSop);
    #ifdef DEBUG
    assert(nVars == faninIds.size());
    #endif

    ll nFrame = smlt.GetFrameNumb();
    dynamic_bitset <ull> product(nFrame, 0);
    for (char * pCube = pSop; *pCube; pCube += nVars + 3) {
        bool isFirst = true;
        for (ll i = 0; pCube[i] != ' '; i++) {
            ll faninId = faninIds[i];
            switch (pCube[i]) {
                case '-':
                    continue;
                    break;
                case '0':
                    if (isFirst) {
                        isFirst = false;
                        product = ~(*smlt.GetDat(faninId));
                    }
                    else
                        product &= ~(*smlt.GetDat(faninId));
                    break;
                case '1':
                    if (isFirst) {
                        isFirst = false;
                        product = (*smlt.GetDat(faninId));
                    }
                    else
                        product &= (*smlt.GetDat(faninId));
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
            value = product;
        else
            value |= product;
    }
}


static bigInt GetValue(vector<dynamic_bitset<ull>> & dat, ll iPatt, bool isSign, ll msb) {
    ll lsb = 0;
    ll shift = msb - lsb;
    bigInt ret(0);
    for (ll k = msb; k >= lsb; --k) {
        ret <<= 1;
        if (dat[k][iPatt])
            ++ret;
    }
    if (isSign && ret >= (bigInt(1) << shift))
        ret = -((bigInt(1) << (shift + 1)) - ret);
    return ret;
}

static vector<bigInt> GetValue_complex(vector<dynamic_bitset<ull>> & dat, ll iPatt, bool isSign, ll msb, ll nOutput) {
    ll lsb = 0;
    ll newmsb = (msb + 1)/nOutput - 1;
    ll shift = newmsb - lsb;
    ll size = shift + 1;
    vector<bigInt> ret(nOutput,0);
    for (auto divnum = 0; divnum < nOutput; ++divnum){
        for (ll k = newmsb + divnum * size; k >= lsb + divnum * size; --k) {
            ret[divnum] <<= 1;
            if (dat[k][iPatt])
                ++ret[divnum];
        }
        if (isSign && ret[divnum] >= (bigInt(1) << shift))
            ret[divnum] = -((bigInt(1) << (shift + 1)) - ret[divnum]);
    }
    return ret;
}

static void CalcSomeLACErr_complex(Simulator & appSmlt, LACMan & lacMan, std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2Nodes, vector<vector<bigInt>> & YAcc, vector<bigInt> & oldErr, bigInt & runMin, timer::progress_display & pd, ll startIndex, ll endIndex, ll nFrame, bool isSign, ll nOutput, METR_TYPE metrType, LAC_TYPE lacType, bool checkCONS, bool useAppDisjCut, vector <ll> RealCom) {
    ll nPo = appSmlt.GetPoNum();
    ll nLargeFrame = appSmlt.GetFrameNumb();
    assert(nFrame <= nLargeFrame);
    vector<bigInt> L2N(nFrame + 1, 0);
    vector<bigInt> sumNegLoss(nFrame + 1, 0);
    vector <dynamic_bitset<ull>> tempOutps(nPo);
    ll oldTargId = -1;

    for (ll lacId = startIndex; lacId < endIndex; ++lacId) {
        auto pLac = lacMan.GetLac(lacId);
        ll targId = pLac->GetTargId();

        // calculate $\partial L / \partial n$
        if (oldTargId != targId) {
            for (ll j = 0; j < nPo; ++j) {
                auto poId = appSmlt.GetPoId(j);
                tempOutps[j] = *appSmlt.GetDat(poId) ^ bdPo2Nodes[j][targId]; 
            }
            sumNegLoss[nFrame] = 0;
            L2N[nFrame] = 0;
            if (metrType == METR_TYPE::MSE) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    auto YNew = GetValue_complex(tempOutps, iPatt, isSign, nPo - 1, nOutput);
                    bigInt newDiff = 0;
                    for (auto i = 0; i < nOutput; i++){
                        YNew[i] += RealCom[i];
                        newDiff += (YNew[i] - YAcc[iPatt][i])*(YNew[i] - YAcc[iPatt][i]);
                    }
                    L2N[iPatt] = (newDiff - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else if (metrType == METR_TYPE::ER) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    auto YNew = GetValue_complex(tempOutps, iPatt, isSign, nPo - 1, nOutput);
                    bool flag = false;  // true: not equal
                    for (auto i = 0; i < nOutput; i++) {
                        YNew[i] += RealCom[i];
                        if (YNew[i] != YAcc[iPatt][i]) {
                            flag = true;
                            break;
                        }
                    }
                    ll newEr;
                    if (flag)
                        newEr = 1;
                    else
                        newEr = 0;
                    L2N[iPatt] = (newEr - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else
                assert(0);
        }
        oldTargId = targId;

        // calculate $\partial n / \partial LAC$
        boost::dynamic_bitset <ull> isChanged(nLargeFrame, 0);
        if (lacType == LAC_TYPE::CONS) {
            auto & specLac = *dynamic_pointer_cast <ConstLAC>(pLac);
            bool isConst0 = specLac.IsConst0();
            if (isConst0) isChanged.reset(); else isChanged.set();
            isChanged ^= (*appSmlt.GetDat(targId));
        }
        else if (lacType == LAC_TYPE::SASIMI) {
            auto & specLac = *dynamic_pointer_cast <SasimiLAC>(pLac);
            ll subId = specLac.GetSubId();
            bool isInv = specLac.GetIsInv();
            isChanged = isInv? ~(*appSmlt.GetDat(targId)) ^ (*appSmlt.GetDat(subId)): (*appSmlt.GetDat(targId)) ^ (*appSmlt.GetDat(subId));
        }
        else if (lacType == LAC_TYPE::RAC) {
            auto & specLac = *dynamic_pointer_cast <RacLAC>(pLac);
            auto divIds = specLac.GetDivIds(); 
            auto sop = specLac.GetSop();
            dynamic_bitset <ull> newValue(nLargeFrame, 0);
            GetNewValue(appSmlt, divIds, sop, newValue);
            isChanged = (*appSmlt.GetDat(targId)) ^ newValue;
        }
        else
            assert(0);

        // calculate error
        bigInt ser = 0;
        if (nFrame == nLargeFrame) {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                ser += (int)isChanged[iPatt] * L2N[iPatt];
                // Acceleration technique 4: begin
                if (ser + sumNegLoss[iPatt] > runMin)
                    break;
                // Acceleration technique 4: end                    
            }
            std::unique_lock<std::mutex> lock(mtx);
            runMin = min(runMin, ser);
            ++pd;
            lock.unlock();
        }
        else {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                ser += (int)isChanged[iPatt] * L2N[iPatt];
                if (useAppDisjCut){
                    if (ser + sumNegLoss[iPatt] > 2 * runMin)
                        break;    
                }
            
            }
            std::unique_lock<std::mutex> lock(mtx);
            runMin = min(runMin, ser);
            ++pd;
            lock.unlock();
        }
        
        pLac->SetErrPro(ser);
    }
}


static void CalcSomeLACErr(Simulator & appSmlt, LACMan & lacMan, std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2Nodes, vector<bigInt> & YAcc, vector<bigInt> & oldErr, bigInt & runMin, timer::progress_display & pd, ll startIndex, ll endIndex, ll nFrame, bool isSign, METR_TYPE metrType, LAC_TYPE lacType, bool useAppDisjCut, vector <ll> RealCom) {
    ll nPo = appSmlt.GetPoNum();
    ll nLargeFrame = appSmlt.GetFrameNumb();
    assert(nFrame <= nLargeFrame);
    vector<bigInt> L2N(nFrame + 1, 0);
    vector<bigInt> sumNegLoss(nFrame + 1, 0);
    vector <dynamic_bitset<ull>> tempOutps(nPo);
    ll oldTargId = -1;

    for (ll lacId = startIndex; lacId < endIndex; ++lacId) {
        auto pLac = lacMan.GetLac(lacId);
        ll targId = pLac->GetTargId();

        // calculate $\partial L / \partial n$
        if (oldTargId != targId) {
            for (ll j = 0; j < nPo; ++j) {
                auto poId = appSmlt.GetPoId(j);
                tempOutps[j] = *appSmlt.GetDat(poId) ^ bdPo2Nodes[j][targId]; 
            }
            sumNegLoss[nFrame] = 0;
            L2N[nFrame] = 0;
            if (metrType == METR_TYPE::MSE) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    auto YNew = GetValue(tempOutps, iPatt, isSign, nPo - 1) + RealCom[0];
                    auto newDiff = (YNew - YAcc[iPatt]);
                    L2N[iPatt] = (newDiff * newDiff - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else if (metrType == METR_TYPE::ER) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    auto YNew = GetValue(tempOutps, iPatt, isSign, nPo - 1) + RealCom[0];
                    auto newDiff = (YNew - YAcc[iPatt]);
                    ll newEr;
                    if (newDiff != 0)
                        newEr = 1;
                    else
                        newEr = 0;
                    L2N[iPatt] = (newEr - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else if (metrType == METR_TYPE::MED) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    auto YNew = GetValue(tempOutps, iPatt, isSign, nPo - 1) + RealCom[0];
                    auto newDiff = abs(YNew - YAcc[iPatt]);
                    L2N[iPatt] = (newDiff - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            // else if (metrType == METR_TYPE::MRED) {
            //     for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
            //         auto YNew = GetValue(tempOutps, iPatt, isSign, nPo - 1) + RealCom[0];
            //         bigFlt newErr;
            //         if (YAcc[iPatt] != 0)
            //             newErr = abs(1 - static_cast <bigFlt>(YNew)/static_cast <bigFlt>(YAcc[iPatt]));
            //         else
            //             newErr = abs(1 - static_cast <bigFlt>(YNew));
            //         L2N[iPatt] = (newErr - oldErr[iPatt]);
            //         sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
            //         if (L2N[iPatt + 1] < 0)
            //             sumNegLoss[iPatt] += L2N[iPatt + 1];
            //     }
            // }
            else if (metrType == METR_TYPE::MHD) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    boost::dynamic_bitset<ull> accPo = bigIntToBin(YAcc[iPatt], nPo, isSign);
                    ll hd = 0;
                    for (ll o = 0; o < nPo; ++o) {
                        // auto poId = appSmlt.GetPoId(o);
                        // if (accPo[o] != appSmlt.GetDat(poId, iPatt))
                        //     ++hd;
                        if (accPo[o] != tempOutps[o][iPatt])
                            ++hd;
                    }
                    L2N[iPatt] = (hd - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else
                assert(0);
        }
        oldTargId = targId;

        // calculate $\partial n / \partial LAC$
        boost::dynamic_bitset <ull> isChanged(nLargeFrame, 0);
        if (lacType == LAC_TYPE::CONS) {
            auto & specLac = *dynamic_pointer_cast <ConstLAC>(pLac);
            bool isConst0 = specLac.IsConst0();
            if (isConst0) isChanged.reset(); else isChanged.set();
            isChanged ^= (*appSmlt.GetDat(targId));
        }
        else if (lacType == LAC_TYPE::SASIMI) {
            auto & specLac = *dynamic_pointer_cast <SasimiLAC>(pLac);
            ll subId = specLac.GetSubId();
            bool isInv = specLac.GetIsInv();
            isChanged = isInv? ~(*appSmlt.GetDat(targId)) ^ (*appSmlt.GetDat(subId)): (*appSmlt.GetDat(targId)) ^ (*appSmlt.GetDat(subId));
        }
        else if (lacType == LAC_TYPE::RAC) {
            auto & specLac = *dynamic_pointer_cast <RacLAC>(pLac);
            auto divIds = specLac.GetDivIds(); 
            auto sop = specLac.GetSop();
            dynamic_bitset <ull> newValue(nLargeFrame, 0);
            GetNewValue(appSmlt, divIds, sop, newValue);
            isChanged = (*appSmlt.GetDat(targId)) ^ newValue;
        }
        else
            assert(0);

        // calculate error
        bigInt ser = 0;
        if (nFrame == nLargeFrame) {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                ser += (int)isChanged[iPatt] * L2N[iPatt];
                // Acceleration technique 4: begin
                if (ser + sumNegLoss[iPatt] > runMin)
                    break;
                // Acceleration technique 4: end                   
            }
            std::unique_lock<std::mutex> lock(mtx);
            runMin = min(runMin, ser);
            ++pd;
            lock.unlock();
        }
        else {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                ser += (int)isChanged[iPatt] * L2N[iPatt];
                if(iPatt > 0){
                    if (useAppDisjCut){
                        if (ser + sumNegLoss[iPatt] > runMin)
                            break; 
                    }
                }            
            }
            std::unique_lock<std::mutex> lock(mtx);
            runMin = min(runMin, ser);
            ++pd;
            lock.unlock();
        }

        // if (metrType == METR_TYPE::MRED)
        //     pLac->SetErrBigFlt(ser);
        // else {
        //     bigInt serBigInt = bigInt(ser);     // attention!!! for MRED, flt to int!
        //     pLac->SetErrPro(serBigInt);
        // }
        pLac->SetErrPro(ser);
    }
}


static void CalcSomeLACErr_MRED(Simulator & appSmlt, LACMan & lacMan, std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2Nodes, vector<bigInt> & YAcc, vector<bigFlt> & oldErr, bigFlt & runMin, timer::progress_display & pd, ll startIndex, ll endIndex, ll nFrame, bool isSign, METR_TYPE metrType, LAC_TYPE lacType, bool useAppDisjCut, vector <ll> RealCom) {
    ll nPo = appSmlt.GetPoNum();
    ll nLargeFrame = appSmlt.GetFrameNumb();
    assert(nFrame <= nLargeFrame);
    vector<bigFlt> L2N(nFrame + 1, 0);
    vector<bigFlt> sumNegLoss(nFrame + 1, 0);
    vector <dynamic_bitset<ull>> tempOutps(nPo);
    ll oldTargId = -1;

    for (ll lacId = startIndex; lacId < endIndex; ++lacId) {
        auto pLac = lacMan.GetLac(lacId);
        ll targId = pLac->GetTargId();

        // calculate $\partial L / \partial n$
        if (oldTargId != targId) {
            for (ll j = 0; j < nPo; ++j) {
                auto poId = appSmlt.GetPoId(j);
                tempOutps[j] = *appSmlt.GetDat(poId) ^ bdPo2Nodes[j][targId]; 
            }
            sumNegLoss[nFrame] = 0;
            L2N[nFrame] = 0;
        
            if (metrType == METR_TYPE::MRED) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    auto YNew = GetValue(tempOutps, iPatt, isSign, nPo - 1) + RealCom[0];
                    bigFlt newErr;
                    if (YAcc[iPatt] != 0)
                        newErr = abs(1 - static_cast <bigFlt>(YNew)/static_cast <bigFlt>(YAcc[iPatt]));
                    else
                        // newErr = abs(1 - static_cast <bigFlt>(YNew));
                        newErr = abs(static_cast <bigFlt>(YNew));
                    L2N[iPatt] = (newErr - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else
                assert(0);
        }
        oldTargId = targId;

        // calculate $\partial n / \partial LAC$
        boost::dynamic_bitset <ull> isChanged(nLargeFrame, 0);
        if (lacType == LAC_TYPE::CONS) {
            auto & specLac = *dynamic_pointer_cast <ConstLAC>(pLac);
            bool isConst0 = specLac.IsConst0();
            if (isConst0) isChanged.reset(); else isChanged.set();
            isChanged ^= (*appSmlt.GetDat(targId));
        }
        else if (lacType == LAC_TYPE::SASIMI) {
            auto & specLac = *dynamic_pointer_cast <SasimiLAC>(pLac);
            ll subId = specLac.GetSubId();
            bool isInv = specLac.GetIsInv();
            isChanged = isInv? ~(*appSmlt.GetDat(targId)) ^ (*appSmlt.GetDat(subId)): (*appSmlt.GetDat(targId)) ^ (*appSmlt.GetDat(subId));
        }
        else if (lacType == LAC_TYPE::RAC) {
            auto & specLac = *dynamic_pointer_cast <RacLAC>(pLac);
            auto divIds = specLac.GetDivIds(); 
            auto sop = specLac.GetSop();
            dynamic_bitset <ull> newValue(nLargeFrame, 0);
            GetNewValue(appSmlt, divIds, sop, newValue);
            isChanged = (*appSmlt.GetDat(targId)) ^ newValue;
        }
        else
            assert(0);

        // calculate error
        bigFlt ser = 0;
        if (nFrame == nLargeFrame) {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                ser += (int)isChanged[iPatt] * L2N[iPatt];
                // Acceleration technique 4: begin
                if (ser + sumNegLoss[iPatt] > runMin)
                    break;
                // Acceleration technique 4: end                   
            }
            std::unique_lock<std::mutex> lock(mtx);
            runMin = min(runMin, ser);
            ++pd;
            lock.unlock();
        }
        else {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                ser += (int)isChanged[iPatt] * L2N[iPatt];
                if(iPatt > 0){
                    if (useAppDisjCut){
                        if (ser + sumNegLoss[iPatt] > runMin)
                            break; 
                    }
                }            
            }
            std::unique_lock<std::mutex> lock(mtx);
            runMin = min(runMin, ser);
            ++pd;
            lock.unlock();
        }

        pLac->SetErrBigFlt(ser);
    }
}



static void CalcSomeLACErr_GetLacPerNode(Simulator & appSmlt, LACMan & lacMan, std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2Nodes, vector<bigInt> & YAcc, vector<bigFlt> & oldErr, bigFlt & runMin, timer::progress_display & pd, ll startIndex, ll endIndex, ll nFrame, bool isSign, METR_TYPE metrType, LAC_TYPE lacType, bool useAppDisjCut, vector <ll> RealCom, unordered_map<ll, bigFlt> & nodeSmallestErr) {
    ll nPo = appSmlt.GetPoNum();
    ll nLargeFrame = appSmlt.GetFrameNumb();
    assert(nFrame <= nLargeFrame);
    vector<bigFlt> L2N(nFrame + 1, 0);
    vector<bigFlt> sumNegLoss(nFrame + 1, 0);
    vector <dynamic_bitset<ull>> tempOutps(nPo);
    ll oldTargId = -1;

    for (ll lacId = startIndex; lacId < endIndex; ++lacId) {
        auto pLac = lacMan.GetLac(lacId);
        ll targId = pLac->GetTargId();

        auto it = nodeSmallestErr.find(targId);

        auto pNode = appSmlt.GetObj(targId);
        bool isConst0LAC = true;    // false: const1 LAC

        // calculate $\partial L / \partial n$
        if (oldTargId != targId) {
            for (ll j = 0; j < nPo; ++j) {
                auto poId = appSmlt.GetPoId(j);
                tempOutps[j] = *appSmlt.GetDat(poId) ^ bdPo2Nodes[j][targId]; 
            }
            sumNegLoss[nFrame] = 0;
            L2N[nFrame] = 0;
            if (metrType == METR_TYPE::MSE) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    auto YNew = GetValue(tempOutps, iPatt, isSign, nPo - 1) + RealCom[0];
                    auto newDiff = (YNew - YAcc[iPatt]);
                    L2N[iPatt] = (bigFlt(newDiff * newDiff) - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else if (metrType == METR_TYPE::ER) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    auto YNew = GetValue(tempOutps, iPatt, isSign, nPo - 1) + RealCom[0];
                    auto newDiff = (YNew - YAcc[iPatt]);
                    bigFlt newEr;
                    if (newDiff != 0)
                        newEr = 1;
                    else
                        newEr = 0;
                    L2N[iPatt] = (newEr - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else if (metrType == METR_TYPE::MED) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    auto YNew = GetValue(tempOutps, iPatt, isSign, nPo - 1) + RealCom[0];
                    auto newDiff = abs(YNew - YAcc[iPatt]);
                    L2N[iPatt] = (bigFlt(newDiff) - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else if (metrType == METR_TYPE::MRED) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    auto YNew = GetValue(tempOutps, iPatt, isSign, nPo - 1) + RealCom[0];
                    bigFlt newErr;
                    if (YAcc[iPatt] != 0)
                        newErr = abs(1 - static_cast <bigFlt>(YNew)/static_cast <bigFlt>(YAcc[iPatt]));
                    else
                        newErr = abs(1 - static_cast <bigFlt>(YNew));
                    L2N[iPatt] = (newErr - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else
                assert(0);
        }
        oldTargId = targId;

        // calculate $\partial n / \partial LAC$
        boost::dynamic_bitset <ull> isChanged(nLargeFrame, 0);
        if (lacType == LAC_TYPE::CONS) {
            auto & specLac = *dynamic_pointer_cast <ConstLAC>(pLac);
            bool isConst0 = specLac.IsConst0();
            if (isConst0) isChanged.reset(); else isChanged.set();
            isChanged ^= (*appSmlt.GetDat(targId));

            if (isConst0)
                isConst0LAC = true;
            else
                isConst0LAC = false;
        }
        else if (lacType == LAC_TYPE::SASIMI) {
            auto & specLac = *dynamic_pointer_cast <SasimiLAC>(pLac);
            ll subId = specLac.GetSubId();
            bool isInv = specLac.GetIsInv();
            isChanged = isInv? ~(*appSmlt.GetDat(targId)) ^ (*appSmlt.GetDat(subId)): (*appSmlt.GetDat(targId)) ^ (*appSmlt.GetDat(subId));
        }
        else if (lacType == LAC_TYPE::RAC) {
            auto & specLac = *dynamic_pointer_cast <RacLAC>(pLac);
            auto divIds = specLac.GetDivIds(); 
            auto sop = specLac.GetSop();
            dynamic_bitset <ull> newValue(nLargeFrame, 0);
            GetNewValue(appSmlt, divIds, sop, newValue);
            isChanged = (*appSmlt.GetDat(targId)) ^ newValue;
        }
        else
            assert(0);

        if (nFrame == nLargeFrame) {
            if (isConst0LAC)
                pNode->fUseSmallFrame0 = 0;
            else
                pNode->fUseSmallFrame1 = 0;
        }
        else {
            if (isConst0LAC)
                pNode->fUseSmallFrame0 = 1;
            else
                pNode->fUseSmallFrame1 = 1;
        }

        // calculate error
        bigFlt ser = 0;
        if (isConst0LAC)
            pNode->fEarlyTerm0 = 0;
        else
            pNode->fEarlyTerm1 = 0;
        if (nFrame == nLargeFrame) {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                ser += (int)isChanged[iPatt] * L2N[iPatt];
                // Acceleration technique 4: begin
                // // if (ser + sumNegLoss[iPatt] > bigFlt(runMin))
                // if (ser + sumNegLoss[iPatt] > it->second) {
                //     // ser = runMin;
                //     if (isConst0LAC)
                //         pNode->fEarlyTerm0 = 1;
                //     else
                //         pNode->fEarlyTerm1 = 1;
                    
                //     break;
                // }
                // Acceleration technique 4: end                   
            }
            // std::unique_lock<std::mutex> lock(mtx);
            // runMin = min(runMin, ser);
            // ++pd;
            // lock.unlock();
        }
        else {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                ser += (int)isChanged[iPatt] * L2N[iPatt];
                // if(iPatt > 0){
                //     if (useAppDisjCut){
                //         // if (ser + sumNegLoss[iPatt] > bigFlt(runMin))
                //         if (ser + sumNegLoss[iPatt] > it->second) {
                //             // ser = runMin;
                //             if (isConst0LAC)
                //                 pNode->fEarlyTerm0 = 1;
                //             else
                //                 pNode->fEarlyTerm1 = 1;

                //             break;
                //         }
                //     }
                // }            
            }
            // std::unique_lock<std::mutex> lock(mtx);
            // runMin = min(runMin, ser);
            // ++pd;
            // lock.unlock();
        }

        if (it != nodeSmallestErr.end()) {
            if (ser < it->second) {
                std::unique_lock<std::mutex> lock(mtx);
                it->second = ser;
                lock.unlock();
            }
            if (abs(ser - it->second)/max(ser, it->second) < 0.2)
                pNode->fMarkF = 1;  // the error of LAC0 and LAC1 are about the same 
        }
        else 
            assert(0);
        
        bigInt serBigInt = bigInt(ser);     // attention!!! for MRED, flt to int!
        pLac->SetErrPro(serBigInt);

        if (metrType == METR_TYPE::ER) {
            if (isConst0LAC)
                pNode->error0 = static_cast<int>(serBigInt);
            else
                pNode->error1 = static_cast<int>(serBigInt);
        }
        else {
            if (isConst0LAC)
                pNode->error0f = double(ser / bigFlt(nFrame)); // error increase
            else
                pNode->error1f = double(ser / bigFlt(nFrame));
        }
    }
}

// Note: for nOutput != 1, still only support MSE metric!!!
void VECBEEMan::CalcLACErrsPlus(Simulator & accSmlt, Simulator & appSmlt, LACMan & lacMan, const bigInt & uppBound, ll nOutput, bool useAppDisjCut, vector <ll> RealCom) {
    cout << "calculating errors of LACs" << endl;
    assert(IsPIOSame(accSmlt, appSmlt));
    assert((nFrame & 63) == 0);
    assert(uppBound >= 0);
    ll nPo = appSmlt.GetPoNum();

    if (nOutput == 1) {
        if (metrType != METR_TYPE::MRED) {
            vector<bigInt> YAcc(nFrame, 0);
            vector<bigInt> Y(nFrame, 0);
            vector<bigInt> oldErr(nFrame, 0);
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                YAcc[iPatt] = accSmlt.GetOutpPro(iPatt, isSign);
                Y[iPatt] = appSmlt.GetOutpPro(iPatt, isSign) + RealCom[0];
            }
            if (metrType == METR_TYPE::MSE) {
                for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                    auto diff = (Y[iPatt] - YAcc[iPatt]);
                    oldErr[iPatt] = diff * diff;
                }
            }
            else if (metrType == METR_TYPE::ER) {
                for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                    if (Y[iPatt] != YAcc[iPatt])
                        oldErr[iPatt] = 1;
                    else
                        oldErr[iPatt] = 0;
                }
            }
            else if (metrType == METR_TYPE::MED) {
                for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                    oldErr[iPatt] = abs(Y[iPatt] - YAcc[iPatt]);
                }
            }
            else if (metrType == METR_TYPE::MHD) {
                for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                    ll hd = 0;
                    for (ll o = 0; o < nPo; ++o) {
                        auto poId = appSmlt.GetPoId(o);
                        if (accSmlt.GetDat(poId, iPatt) != appSmlt.GetDat(poId, iPatt))
                            ++hd;
                    }
                    oldErr[iPatt] = hd;
                }
            }
            else 
                assert(0);
            cout << "calculating LAC errors" << endl;
            {
                ll lacNum = lacMan.GetLacNum();
                ll realThread = min(nThread, lacNum);
                assert(realThread > 0);
                cout << "real thread number: " << realThread << endl;
                ll chunkSize = lacNum / realThread;
                ll remainder = lacNum % realThread;
                timer::progress_display pd(lacMan.GetLacNum());
                // bigFlt runMin = bigFlt(uppBound + 1);
                bigInt runMin = uppBound + 1;
                const ll smallFrame = 10048;
                assert(smallFrame <= nFrame);
                vector<thread> threads;
                ll start = 0;
                for (ll i = 0; i < realThread; ++i) {
                    ll end = start + chunkSize + (i < remainder? 1: 0);
                    threads.emplace_back(CalcSomeLACErr, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMin), std::ref(pd), start, end, smallFrame, isSign, metrType, lacType, useAppDisjCut, RealCom);
                    start = end;
                }
                for (auto& thread: threads)
                    thread.join();
            }
            // collect promising LACs
            const double perc_large = 0.1;
            const double perc_small = 0.0001;
            if (lacType == LAC_TYPE::SASIMI && useAppDisjCut){
                lacMan.Filt(perc_small);
                cout << "Filt = " <<  perc_small << endl;
            }
            else{
                lacMan.Filt(perc_large);
                cout << "Filt = " <<  perc_large << endl;
            }
            {
                ll lacNum = lacMan.GetLacNum();
                cout << "lacNum(after filtering) = " << lacNum << endl;
                ll realThread = min(nThread, lacNum);
                assert(realThread > 0);
                cout << "real thread number: " << realThread << endl;
                ll chunkSize = lacNum / realThread;
                ll remainder = lacNum % realThread;
                timer::progress_display pd(lacMan.GetLacNum());
                // bigFlt runMin = bigFlt(uppBound + 1);
                bigInt runMin = uppBound + 1;
                vector<thread> threads;
                ll start = 0;
                for (ll i = 0; i < realThread; ++i) {
                    ll end = start + chunkSize + (i < remainder? 1: 0);
                    threads.emplace_back(CalcSomeLACErr, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMin), std::ref(pd), start, end, nFrame, isSign, metrType, lacType, useAppDisjCut, RealCom);
                    start = end;
                }
                for (auto& thread: threads)
                    thread.join();
            }
        }
        else {      // MRED
            vector<bigInt> YAcc(nFrame, 0);
            vector<bigInt> Y(nFrame, 0);
            vector<bigFlt> oldErr(nFrame, 0);
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                YAcc[iPatt] = accSmlt.GetOutpPro(iPatt, isSign);
                Y[iPatt] = appSmlt.GetOutpPro(iPatt, isSign) + RealCom[0];
            }         

            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                if (YAcc[iPatt] != 0)
                    oldErr[iPatt] = abs(1 - static_cast <bigFlt>(Y[iPatt])/static_cast <bigFlt>(YAcc[iPatt]));
                else
                    // oldErr[iPatt] = abs(1 - static_cast <bigFlt>(Y[iPatt]));
                    oldErr[iPatt] = abs(static_cast <bigFlt>(Y[iPatt]));
            }

            cout << "calculating LAC errors" << endl;
            {
                ll lacNum = lacMan.GetLacNum();
                ll realThread = min(nThread, lacNum);
                assert(realThread > 0);
                cout << "real thread number: " << realThread << endl;
                ll chunkSize = lacNum / realThread;
                ll remainder = lacNum % realThread;
                timer::progress_display pd(lacMan.GetLacNum());
                bigFlt runMin = bigFlt(uppBound + 1);
                // bigInt runMin = uppBound + 1;
                const ll smallFrame = 10048;
                assert(smallFrame <= nFrame);
                vector<thread> threads;
                ll start = 0;
                for (ll i = 0; i < realThread; ++i) {
                    ll end = start + chunkSize + (i < remainder? 1: 0);
                    threads.emplace_back(CalcSomeLACErr_MRED, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMin), std::ref(pd), start, end, smallFrame, isSign, metrType, lacType, useAppDisjCut, RealCom);
                    start = end;
                }
                for (auto& thread: threads)
                    thread.join();
            }
            // collect promising LACs
            const double perc = 0.5;
            lacMan.Filt(perc);
            cout << "Filt = " <<  perc << endl;
            {
                ll lacNum = lacMan.GetLacNum();
                cout << "lacNum(after filtering) = " << lacNum << endl;
                ll realThread = min(nThread, lacNum);
                assert(realThread > 0);
                cout << "real thread number: " << realThread << endl;
                ll chunkSize = lacNum / realThread;
                ll remainder = lacNum % realThread;
                timer::progress_display pd(lacMan.GetLacNum());
                bigFlt runMin = bigFlt(uppBound + 1);
                // bigInt runMin = uppBound + 1;
                vector<thread> threads;
                ll start = 0;
                for (ll i = 0; i < realThread; ++i) {
                    ll end = start + chunkSize + (i < remainder? 1: 0);
                    threads.emplace_back(CalcSomeLACErr_MRED, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMin), std::ref(pd), start, end, nFrame, isSign, metrType, lacType, useAppDisjCut, RealCom);
                    start = end;
                }
                for (auto& thread: threads)
                    thread.join();
            }
        }
    }
    else {  // nOutput != 1
        vector<vector<bigInt> > YAcc(nFrame, vector<bigInt>(nOutput, 0));
        vector<vector<bigInt> > Y(nFrame, vector<bigInt>(nOutput, 0));
        vector<bigInt> oldErr(nFrame, 0);
        for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
            YAcc[iPatt] = accSmlt.GetOutpPro_complex(iPatt, isSign, nOutput);
            Y[iPatt] = appSmlt.GetOutpPro_complex(iPatt, isSign, nOutput);
            assert(Y[iPatt].size() == nOutput);
        }
        if (metrType == METR_TYPE::MSE) {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                bigInt diff = 0;
                bigInt square = 0;
                for (ll k = 0; k < nOutput; k++){
                    Y[iPatt][k] += RealCom[k];
                    diff = Y[iPatt][k] - YAcc[iPatt][k];
                    square += diff * diff;
                }
                oldErr[iPatt] = square;
            }
        }
        else if (metrType == METR_TYPE::ER) {         
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                bool flag = false;
                for (ll k = 0; k < nOutput; k++) {
                    if (Y[iPatt][k] != YAcc[iPatt][k]) {
                        flag = true;
                        break;
                    }
                }
                if (flag)
                    oldErr[iPatt] = 1;
                else
                    oldErr[iPatt] = 0;
            }
        }
        cout << "calculating LAC errors" << endl;
        {
            ll lacNum = lacMan.GetLacNum();
            ll realThread = min(nThread, lacNum);
            assert(realThread > 0);
            cout << "real thread number: " << realThread << endl;
            ll chunkSize = lacNum / realThread;
            ll remainder = lacNum % realThread;
            timer::progress_display pd(lacMan.GetLacNum());
            bigInt runMin = uppBound + 1;
            const ll smallFrame = 10048;
            assert(smallFrame <= nFrame);
            vector<thread> threads;
            ll start = 0;
            for (ll i = 0; i < realThread; ++i) {
                ll end = start + chunkSize + (i < remainder? 1: 0);
                threads.emplace_back(CalcSomeLACErr_complex, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMin), std::ref(pd), start, end, smallFrame, isSign, nOutput, metrType, lacType, 0, useAppDisjCut, RealCom);
                start = end;
            }
            for (auto& thread: threads)
                thread.join();
        }
        // collect promising LACs
        const double perc_large = 0.1;
        const double perc_small = 0.0001;
        if (lacType == LAC_TYPE::SASIMI && useAppDisjCut){
            lacMan.Filt(perc_small);
            cout << "Filt = " <<  perc_small << endl;
        }
        else{
            lacMan.Filt(perc_large);
            cout << "Filt = " <<  perc_large << endl;
        }
        {
            ll lacNum = lacMan.GetLacNum();
            ll realThread = min(nThread, lacNum);
            assert(realThread > 0);
            cout << "real thread number: " << realThread << endl;
            ll chunkSize = lacNum / realThread;
            ll remainder = lacNum % realThread;
            timer::progress_display pd(lacMan.GetLacNum());
            bigInt runMin = uppBound + 1;
            vector<thread> threads;
            ll start = 0;
            for (ll i = 0; i < realThread; ++i) {
                ll end = start + chunkSize + (i < remainder? 1: 0);
                threads.emplace_back(CalcSomeLACErr_complex, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMin), std::ref(pd), start, end, nFrame, isSign, nOutput, metrType, lacType, 0, useAppDisjCut, RealCom);
                start = end;
            }
            for (auto& thread: threads)
                thread.join();
        }
    }
}



// Note: for nOutput != 1, still only support MSE metric!!!
void VECBEEMan::CalcLACErrsPlus_GetLacPerNode(Simulator & accSmlt, Simulator & appSmlt, LACMan & lacMan, const bigInt & uppBound, ll nOutput, bool useAppDisjCut, vector <ll> RealCom, unordered_map <ll, std::shared_ptr <LAC>> & LacPerNode, ll idMaxPlus1, bool fFilt, NetMan & net) {
    cout << "begin calculating errors of LACs: " << endl;
    assert(IsPIOSame(accSmlt, appSmlt));
    assert((nFrame & 63) == 0);
    assert(uppBound >= 0);
    // ll nPo = appSmlt.GetPoNum();

    bigFlt runMinNew = bigFlt(uppBound + (uppBound >> 4)); // uppBound * 1.0625
    unordered_map<ll, bigFlt> nodeSmallestErr;
    for (ll i = 1; i < idMaxPlus1; ++i) {
        nodeSmallestErr[i] = runMinNew;
    }

    // clean all fMarkF
    Abc_Obj_t * pObj;
    ll i;
    Abc_NtkForEachObj(net.GetNet(), pObj, i)
        pObj->fMarkF = 0;
    
    if (nOutput == 1) {
        vector<bigInt> YAcc(nFrame, 0);
        vector<bigInt> Y(nFrame, 0);
        vector<bigFlt> oldErr(nFrame, 0);
        for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
            YAcc[iPatt] = accSmlt.GetOutpPro(iPatt, isSign);
            Y[iPatt] = appSmlt.GetOutpPro(iPatt, isSign) + RealCom[0];
        }
        if (metrType == METR_TYPE::MSE) {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                auto diff = (Y[iPatt] - YAcc[iPatt]);
                oldErr[iPatt] = bigFlt(diff * diff);
            }
        }
        else if (metrType == METR_TYPE::ER) {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                if (Y[iPatt] != YAcc[iPatt])
                    oldErr[iPatt] = 1;
                else
                    oldErr[iPatt] = 0;
            }
        }
        else if (metrType == METR_TYPE::MED) {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                oldErr[iPatt] = bigFlt(abs(Y[iPatt] - YAcc[iPatt]));
            }
        }
        else if (metrType == METR_TYPE::MRED) {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                if (YAcc[iPatt] != 0)
                    oldErr[iPatt] = abs(1 - static_cast <bigFlt>(Y[iPatt])/static_cast <bigFlt>(YAcc[iPatt]));
                else
                    oldErr[iPatt] = abs(1 - static_cast <bigFlt>(Y[iPatt]));
            }
        }
        else 
            assert(0);

        if (!fFilt) {
            cout << "calculate all LACs' error using big nFrame(" << nFrame << "): " << endl;
            {
                ll lacNum = lacMan.GetLacNum();
                cout << "#LACs = " << lacNum << endl;
                ll realThread = min(nThread, lacNum);
                assert(realThread > 0);
                cout << "real thread number: " << realThread << endl;
                ll chunkSize = lacNum / realThread;
                ll remainder = lacNum % realThread;
                timer::progress_display pd(lacMan.GetLacNum());
                vector<thread> threads;
                ll start = 0;
                for (ll i = 0; i < realThread; ++i) {
                    ll end = start + chunkSize + (i < remainder? 1: 0);
                    threads.emplace_back(CalcSomeLACErr_GetLacPerNode, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMinNew), std::ref(pd), start, end, nFrame, isSign, metrType, lacType, useAppDisjCut, RealCom, std::ref(nodeSmallestErr));
                    start = end;
                }
                for (auto& thread: threads)
                    thread.join();
            }
        }
        else {
            const ll smallFrame = 10048;
            cout << "calculate all LACs' error using small nFrame(" << smallFrame << "): " << endl;
            {
                ll lacNum = lacMan.GetLacNum();
                ll realThread = min(nThread, lacNum);
                assert(realThread > 0);
                cout << "real thread number: " << realThread << endl;
                ll chunkSize = lacNum / realThread;
                ll remainder = lacNum % realThread;
                timer::progress_display pd(lacMan.GetLacNum());
                vector<thread> threads;
                ll start = 0;
                for (ll i = 0; i < realThread; ++i) {
                    ll end = start + chunkSize + (i < remainder? 1: 0);
                    threads.emplace_back(CalcSomeLACErr_GetLacPerNode, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMinNew), std::ref(pd), start, end, smallFrame, isSign, metrType, lacType, useAppDisjCut, RealCom, std::ref(nodeSmallestErr));
                    start = end;
                }
                for (auto& thread: threads)
                    thread.join();
            }
            // collect promising LACs
            const double perc_large = 0.5;
            const double perc_small = 0.0001;
            if (lacType == LAC_TYPE::SASIMI && useAppDisjCut) {
                lacMan.FiltPro(perc_small, net);
                cout << "Filt = " <<  perc_small << endl;
            }
            else{
                lacMan.FiltPro(perc_large, net);
                cout << "Filt = " <<  perc_large << endl;
            }
            cout << "calculate all LACs' error using big nFrame(" << nFrame << "): " << endl;
            {
                ll lacNum = lacMan.GetLacNum();
                cout << "#LACs = " << lacNum << endl;
                ll realThread = min(nThread, lacNum);
                assert(realThread > 0);
                cout << "real thread number: " << realThread << endl;
                ll chunkSize = lacNum / realThread;
                ll remainder = lacNum % realThread;
                timer::progress_display pd(lacMan.GetLacNum());
                vector<thread> threads;
                ll start = 0;
                for (ll i = 0; i < realThread; ++i) {
                    ll end = start + chunkSize + (i < remainder? 1: 0);
                    threads.emplace_back(CalcSomeLACErr_GetLacPerNode, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMinNew), std::ref(pd), start, end, nFrame, isSign, metrType, lacType, useAppDisjCut, RealCom, std::ref(nodeSmallestErr));
                    start = end;
                }
                for (auto& thread: threads)
                    thread.join();
            }
        }
    }
    else {  // nOutput != 1
        vector<vector<bigInt> > YAcc(nFrame, vector<bigInt>(nOutput, 0));
        vector<vector<bigInt> > Y(nFrame, vector<bigInt>(nOutput, 0));
        vector<bigInt> oldErr(nFrame, 0);
        for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
            YAcc[iPatt] = accSmlt.GetOutpPro_complex(iPatt, isSign, nOutput);
            Y[iPatt] = appSmlt.GetOutpPro_complex(iPatt, isSign, nOutput);
            assert(Y[iPatt].size() == nOutput);
        }
        if (metrType == METR_TYPE::MSE) {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                bigInt diff = 0;
                bigInt square = 0;
                for (ll k = 0; k < nOutput; k++){
                    Y[iPatt][k] += RealCom[k];
                    diff = Y[iPatt][k] - YAcc[iPatt][k];
                    square += diff * diff;
                }
                oldErr[iPatt] = square;
            }
        }
        else if (metrType == METR_TYPE::ER) {         
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                bool flag = false;
                for (ll k = 0; k < nOutput; k++) {
                    if (Y[iPatt][k] != YAcc[iPatt][k]) {
                        flag = true;
                        break;
                    }
                }
                if (flag)
                    oldErr[iPatt] = 1;
                else
                    oldErr[iPatt] = 0;
            }
        }
        cout << "calculate error for all LACs accurately!" << endl;
        {
            ll lacNum = lacMan.GetLacNum();
            ll realThread = min(nThread, lacNum);
            assert(realThread > 0);
            cout << "real thread number: " << realThread << endl;
            ll chunkSize = lacNum / realThread;
            ll remainder = lacNum % realThread;
            timer::progress_display pd(lacMan.GetLacNum());
            bigInt runMin = uppBound + 1;
            vector<thread> threads;
            ll start = 0;
            for (ll i = 0; i < realThread; ++i) {
                ll end = start + chunkSize + (i < remainder? 1: 0);
                threads.emplace_back(CalcSomeLACErr_complex, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMin), std::ref(pd), start, end, nFrame, isSign, nOutput, metrType, lacType, 0, useAppDisjCut, RealCom);
                start = end;
            }
            for (auto& thread: threads)
                thread.join();
        }
    }

    // if (!LacPerNode.empty())
    //     LacPerNode.clear();
    // LacPerNode.resize(idMaxPlus1);
    // for (ll i = 0; i < lacMan.GetLacNum(); ++i) {
    //     auto pLac = lacMan.GetLac(i);
    //     ll targId = pLac->GetTargId();
    //     if (LacPerNode[targId] == nullptr)
    //         LacPerNode[targId] = pLac;
    //     else {
    //         if (LacPerNode[targId]->GetErrPro() > pLac->GetErrPro())    // update LAC with the smallest error for the node
    //             LacPerNode[targId] = pLac;  
    //     }
    // }

    cout << "finish calc LACs!" << endl;
    if (!LacPerNode.empty())
        LacPerNode.clear();
    for (ll i = 0; i < lacMan.GetLacNum(); ++i) {
        auto pLac = lacMan.GetLac(i);
        ll targId = pLac->GetTargId();

        auto it = LacPerNode.find(targId);
        if (it == LacPerNode.end()) {
            LacPerNode[targId] = pLac;
        }
        else {
            if (LacPerNode[targId]->GetErrPro() > pLac->GetErrPro())    // update LAC with the smallest error for the node
                it->second = pLac; 
        }
    }
}

void VECBEEMan::BatchErrEst(NetMan & appNet, Simulator & accSmlt, Simulator & appSmlt, LACMan & lacMan, const bigInt & uppBound, ll nOutput, vector <ll> RealCom, ll nCand) {
    auto topoNodes = appNet.TopoSort();
    FindDisjCut(appNet, topoNodes);

    CalcBoolDiffCut2Node(appSmlt, topoNodes);
    CalcBoolDiffPo2NodePlus(appSmlt, topoNodes);
    CalcLACErrs(accSmlt, appSmlt, lacMan, uppBound, nOutput, RealCom, nCand);
}


static void CalcSomeLACErrPro(Simulator & appSmlt, LACMan & lacMan, std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2Nodes, vector<bigInt> & YAcc, vector<bigInt> & oldErr, bigInt & runMin, timer::progress_display & pd, ll startIndex, ll endIndex, ll nFrame, bool isSign, METR_TYPE metrType, LAC_TYPE lacType, vector <ll> RealCom) {
    ll nPo = appSmlt.GetPoNum();
    ll nLargeFrame = appSmlt.GetFrameNumb();
    assert(nFrame <= nLargeFrame);
    vector<bigInt> L2N(nFrame + 1, 0);
    vector<bigInt> sumNegLoss(nFrame + 1, 0);
    vector <dynamic_bitset<ull>> tempOutps(nPo);
    ll oldTargId = -1;

    for (ll lacId = startIndex; lacId < endIndex; ++lacId) {
        auto pLac = lacMan.GetLac(lacId);
        ll targId = pLac->GetTargId();

        // calculate $\partial L / \partial n$
        if (oldTargId != targId) {
            for (ll j = 0; j < nPo; ++j) {
                auto poId = appSmlt.GetPoId(j);
                tempOutps[j] = *appSmlt.GetDat(poId) ^ bdPo2Nodes[j][targId]; 
            }
            sumNegLoss[nFrame] = 0;
            L2N[nFrame] = 0;
            if (metrType == METR_TYPE::MSE) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    auto YNew = GetValue(tempOutps, iPatt, isSign, nPo - 1) + RealCom[0];
                    auto newDiff = (YNew - YAcc[iPatt]);
                    L2N[iPatt] = (newDiff * newDiff - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else if (metrType == METR_TYPE::ER) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    auto YNew = GetValue(tempOutps, iPatt, isSign, nPo - 1) + RealCom[0];
                    auto newDiff = (YNew - YAcc[iPatt]);
                    ll newEr;
                    if (newDiff != 0)
                        newEr = 1;
                    else
                        newEr = 0;
                    L2N[iPatt] = (newEr - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else if (metrType == METR_TYPE::MED) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    auto YNew = GetValue(tempOutps, iPatt, isSign, nPo - 1) + RealCom[0];
                    auto newDiff = abs(YNew - YAcc[iPatt]);
                    L2N[iPatt] = (newDiff - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else if (metrType == METR_TYPE::MHD) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    boost::dynamic_bitset<ull> accPo = bigIntToBin(YAcc[iPatt], nPo, isSign);
                    ll hd = 0;
                    for (ll o = 0; o < nPo; ++o) {
                        // auto poId = appSmlt.GetPoId(o);
                        // if (accPo[o] != appSmlt.GetDat(poId, iPatt))
                        if (accPo[o] != tempOutps[o][iPatt])
                            ++hd;
                    }
                    L2N[iPatt] = (hd - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else
                assert(0);
        }
        oldTargId = targId;

        // calculate $\partial n / \partial LAC$
        boost::dynamic_bitset <ull> isChanged(nLargeFrame, 0);
        if (lacType == LAC_TYPE::CONS) {
            auto & specLac = *dynamic_pointer_cast <ConstLAC>(pLac);
            bool isConst0 = specLac.IsConst0();
            if (isConst0) isChanged.reset(); else isChanged.set();
            isChanged ^= (*appSmlt.GetDat(targId));
        }
        else if (lacType == LAC_TYPE::SASIMI) {
            auto & specLac = *dynamic_pointer_cast <SasimiLAC>(pLac);
            ll subId = specLac.GetSubId();
            bool isInv = specLac.GetIsInv();
            isChanged = isInv? ~(*appSmlt.GetDat(targId)) ^ (*appSmlt.GetDat(subId)): (*appSmlt.GetDat(targId)) ^ (*appSmlt.GetDat(subId));
        }
        else if (lacType == LAC_TYPE::RAC) {
            auto & specLac = *dynamic_pointer_cast <RacLAC>(pLac);
            auto divIds = specLac.GetDivIds(); 
            auto sop = specLac.GetSop();
            dynamic_bitset <ull> newValue(nLargeFrame, 0);
            GetNewValue(appSmlt, divIds, sop, newValue);
            isChanged = (*appSmlt.GetDat(targId)) ^ newValue;
        }
        else
            assert(0);

        // calculate error
        bigInt ser = 0;
        if (nFrame == nLargeFrame) {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                ser += (int)isChanged[iPatt] * L2N[iPatt];
                if (ser + sumNegLoss[iPatt] > runMin)
                    break;                  
            }
            std::unique_lock<std::mutex> lock(mtx);
            // runMin = min(runMin, ser);
            ++pd;
            lock.unlock();
        }
        else {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                ser += (int)isChanged[iPatt] * L2N[iPatt];
                if(iPatt > 0){
                    if (ser + sumNegLoss[iPatt] > runMin)
                        break; 
                }            
            }
            std::unique_lock<std::mutex> lock(mtx);
            // runMin = min(runMin, ser);
            ++pd;
            lock.unlock();
        }

        pLac->SetErrPro(ser);
    }
}

static void CalcSomeLACErrPro_MRED(Simulator & appSmlt, LACMan & lacMan, std::vector < std::vector < boost::dynamic_bitset <ull> > > & bdPo2Nodes, vector<bigInt> & YAcc, vector<bigFlt> & oldErr, bigFlt & runMin, timer::progress_display & pd, ll startIndex, ll endIndex, ll nFrame, bool isSign, METR_TYPE metrType, LAC_TYPE lacType, vector <ll> RealCom) {
    ll nPo = appSmlt.GetPoNum();
    ll nLargeFrame = appSmlt.GetFrameNumb();
    assert(nFrame <= nLargeFrame);
    vector<bigFlt> L2N(nFrame + 1, 0);
    vector<bigFlt> sumNegLoss(nFrame + 1, 0);
    vector <dynamic_bitset<ull>> tempOutps(nPo);
    ll oldTargId = -1;

    for (ll lacId = startIndex; lacId < endIndex; ++lacId) {
        auto pLac = lacMan.GetLac(lacId);
        ll targId = pLac->GetTargId();

        // calculate $\partial L / \partial n$
        if (oldTargId != targId) {
            for (ll j = 0; j < nPo; ++j) {
                auto poId = appSmlt.GetPoId(j);
                tempOutps[j] = *appSmlt.GetDat(poId) ^ bdPo2Nodes[j][targId]; 
            }
            sumNegLoss[nFrame] = 0;
            L2N[nFrame] = 0;
            if (metrType == METR_TYPE::MRED) {
                for (ll iPatt = nFrame - 1; iPatt >= 0; --iPatt) {
                    auto YNew = GetValue(tempOutps, iPatt, isSign, nPo - 1) + RealCom[0];
                    bigFlt newErr;
                    if (YAcc[iPatt] != 0)
                        newErr = abs(1 - static_cast <bigFlt>(YNew)/static_cast <bigFlt>(YAcc[iPatt]));
                    else
                        newErr = abs(static_cast <bigFlt>(YNew));
                    L2N[iPatt] = (newErr - oldErr[iPatt]);
                    sumNegLoss[iPatt] = sumNegLoss[iPatt + 1];
                    if (L2N[iPatt + 1] < 0)
                        sumNegLoss[iPatt] += L2N[iPatt + 1];
                }
            }
            else
                assert(0);
        }
        oldTargId = targId;

        // calculate $\partial n / \partial LAC$
        boost::dynamic_bitset <ull> isChanged(nLargeFrame, 0);
        if (lacType == LAC_TYPE::CONS) {
            auto & specLac = *dynamic_pointer_cast <ConstLAC>(pLac);
            bool isConst0 = specLac.IsConst0();
            if (isConst0) isChanged.reset(); else isChanged.set();
            isChanged ^= (*appSmlt.GetDat(targId));
        }
        else if (lacType == LAC_TYPE::SASIMI) {
            auto & specLac = *dynamic_pointer_cast <SasimiLAC>(pLac);
            ll subId = specLac.GetSubId();
            bool isInv = specLac.GetIsInv();
            isChanged = isInv? ~(*appSmlt.GetDat(targId)) ^ (*appSmlt.GetDat(subId)): (*appSmlt.GetDat(targId)) ^ (*appSmlt.GetDat(subId));
        }
        else if (lacType == LAC_TYPE::RAC) {
            auto & specLac = *dynamic_pointer_cast <RacLAC>(pLac);
            auto divIds = specLac.GetDivIds(); 
            auto sop = specLac.GetSop();
            dynamic_bitset <ull> newValue(nLargeFrame, 0);
            GetNewValue(appSmlt, divIds, sop, newValue);
            isChanged = (*appSmlt.GetDat(targId)) ^ newValue;
        }
        else
            assert(0);

        // calculate error
        bigFlt ser = 0;
        if (nFrame == nLargeFrame) {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                ser += (int)isChanged[iPatt] * L2N[iPatt];
                if (ser + sumNegLoss[iPatt] > bigFlt(runMin))
                    break;                  
            }
            std::unique_lock<std::mutex> lock(mtx);
            // runMin = min(runMin, ser);
            ++pd;
            lock.unlock();
        }
        else {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                ser += (int)isChanged[iPatt] * L2N[iPatt];
                if(iPatt > 0){
                    if (ser + sumNegLoss[iPatt] > bigFlt(runMin))
                        break; 
                }            
            }
            std::unique_lock<std::mutex> lock(mtx);
            // runMin = min(runMin, ser);
            ++pd;
            lock.unlock();
        }

        pLac->SetErrBigFlt(ser);
    }
}

// Note: for nOutput != 1, still only support MSE metric!!!
void VECBEEMan::CalcLACErrs(Simulator & accSmlt, Simulator & appSmlt, LACMan & lacMan, const bigInt & uppBound, ll nOutput, vector <ll> RealCom, ll nCand) {
    cout << "calculating errors of LACs" << endl;
    assert(IsPIOSame(accSmlt, appSmlt));
    assert((nFrame & 63) == 0);
    assert(uppBound >= 0);
    ll nPo = appSmlt.GetPoNum();

    if (nOutput == 1) {
        if (metrType != METR_TYPE::MRED) {
            vector<bigInt> YAcc(nFrame, 0);
            vector<bigInt> Y(nFrame, 0);
            vector<bigInt> oldErr(nFrame, 0);
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                YAcc[iPatt] = accSmlt.GetOutpPro(iPatt, isSign);
                Y[iPatt] = appSmlt.GetOutpPro(iPatt, isSign) + RealCom[0];
            }
            if (metrType == METR_TYPE::MSE) {
                for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                    auto diff = (Y[iPatt] - YAcc[iPatt]);
                    oldErr[iPatt] = diff * diff;
                }
            }
            else if (metrType == METR_TYPE::ER) {
                for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                    if (Y[iPatt] != YAcc[iPatt])
                        oldErr[iPatt] = 1;
                    else
                        oldErr[iPatt] = 0;
                }
            }
            else if (metrType == METR_TYPE::MED) {
                for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                    oldErr[iPatt] = abs(Y[iPatt] - YAcc[iPatt]);
                }
            }
            else if (metrType == METR_TYPE::MHD) {
                for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                    ll hd = 0;
                    for (ll o = 0; o < nPo; ++o) {
                        auto poId = appSmlt.GetPoId(o);
                        if (accSmlt.GetDat(poId, iPatt) != appSmlt.GetDat(poId, iPatt))
                            ++hd;
                    }
                    oldErr[iPatt] = hd;
                }
            }
            else 
                assert(0);
            
            bool fFilt = false;
            // double perc = static_cast<double> (nCand) / static_cast<double> (lacMan.GetLacNum());
            // if (perc <= 0.3) {
            //     cout << "nCand/#Lac = " << perc;
            //     perc *= 3;
            //     fFilt = true;
            //     cout << ", use small frame to filter with perc = " << perc << endl;
            // }
            // else {
            //     cout << "nCand/#Lac = " << perc << ", do not use small frame to filter" << endl;
            // }
            if (lacMan.GetLacNum() > 20000)     // can be tuned
                fFilt = true;
            double perc = 0.1;
        
            if (fFilt) {
                cout << "calculating LAC errors using smallFrame (10048)" << endl;
                ll lacNum = lacMan.GetLacNum();
                ll realThread = min(nThread, lacNum);
                assert(realThread > 0);
                cout << "real thread number: " << realThread << endl;
                ll chunkSize = lacNum / realThread;
                ll remainder = lacNum % realThread;
                timer::progress_display pd(lacMan.GetLacNum());
                bigInt runMin = uppBound + 1;
                const ll smallFrame = 10048;
                assert(smallFrame <= nFrame);
                vector<thread> threads;
                ll start = 0;
                for (ll i = 0; i < realThread; ++i) {
                    ll end = start + chunkSize + (i < remainder? 1: 0);
                    threads.emplace_back(CalcSomeLACErrPro, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMin), std::ref(pd), start, end, smallFrame, isSign, metrType, lacType, RealCom);
                    start = end;
                }
                for (auto& thread: threads)
                    thread.join();
                
                // collect promising LACs
                lacMan.Filt(perc);
            }

            cout << "calculating LAC errors using nFrame (" << nFrame << ")" << endl;
            {
                ll lacNum = lacMan.GetLacNum();
                cout << "lacNum(after filtering) = " << lacNum << endl;
                ll realThread = min(nThread, lacNum);
                assert(realThread > 0);
                cout << "real thread number: " << realThread << endl;
                ll chunkSize = lacNum / realThread;
                ll remainder = lacNum % realThread;
                timer::progress_display pd(lacMan.GetLacNum());
                bigInt runMin = uppBound + 1;
                vector<thread> threads;
                ll start = 0;
                for (ll i = 0; i < realThread; ++i) {
                    ll end = start + chunkSize + (i < remainder? 1: 0);
                    threads.emplace_back(CalcSomeLACErrPro, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMin), std::ref(pd), start, end, nFrame, isSign, metrType, lacType, RealCom);
                    start = end;
                }
                for (auto& thread: threads)
                    thread.join();
            }
        }
        else {      // metrType is MRED
            vector<bigInt> YAcc(nFrame, 0);
            vector<bigInt> Y(nFrame, 0);
            vector<bigFlt> oldErr(nFrame, 0);
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                YAcc[iPatt] = accSmlt.GetOutpPro(iPatt, isSign);
                Y[iPatt] = appSmlt.GetOutpPro(iPatt, isSign) + RealCom[0];
            }
            
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                if (YAcc[iPatt] != 0)
                    oldErr[iPatt] = abs(1 - static_cast <bigFlt>(Y[iPatt])/static_cast <bigFlt>(YAcc[iPatt]));
                else
                    // oldErr[iPatt] = abs(1 - static_cast <bigFlt>(Y[iPatt]));
                    oldErr[iPatt] = abs(static_cast <bigFlt>(Y[iPatt]));
            }
            
            bool fFilt = false;
            double perc = static_cast<double> (nCand) / static_cast<double> (lacMan.GetLacNum());
            if (perc <= 0.3) {
                cout << "nCand/#Lac = " << perc;
                perc *= 3;
                fFilt = true;
                cout << ", use small frame to filter with perc = " << perc << endl;
            }
            else {
                cout << "nCand/#Lac = " << perc << ", do not use small frame to filter" << endl;
            }
        
            if (fFilt) {
                cout << "calculating LAC errors using smallFrame (10048)" << endl;
                ll lacNum = lacMan.GetLacNum();
                ll realThread = min(nThread, lacNum);
                assert(realThread > 0);
                cout << "real thread number: " << realThread << endl;
                ll chunkSize = lacNum / realThread;
                ll remainder = lacNum % realThread;
                timer::progress_display pd(lacMan.GetLacNum());
                bigFlt runMin = bigFlt(uppBound + 1);
                const ll smallFrame = 10048;
                assert(smallFrame <= nFrame);
                vector<thread> threads;
                ll start = 0;
                for (ll i = 0; i < realThread; ++i) {
                    ll end = start + chunkSize + (i < remainder? 1: 0);
                    threads.emplace_back(CalcSomeLACErrPro_MRED, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMin), std::ref(pd), start, end, smallFrame, isSign, metrType, lacType, RealCom);
                    start = end;
                }
                for (auto& thread: threads)
                    thread.join();
                
                // collect promising LACs
                lacMan.Filt(perc);
            }

            cout << "calculating LAC errors using nFrame (" << nFrame << ")" << endl;
            {
                ll lacNum = lacMan.GetLacNum();
                cout << "lacNum(after filtering) = " << lacNum << endl;
                ll realThread = min(nThread, lacNum);
                assert(realThread > 0);
                cout << "real thread number: " << realThread << endl;
                ll chunkSize = lacNum / realThread;
                ll remainder = lacNum % realThread;
                timer::progress_display pd(lacMan.GetLacNum());
                bigFlt runMin = bigFlt(uppBound + 1);
                vector<thread> threads;
                ll start = 0;
                for (ll i = 0; i < realThread; ++i) {
                    ll end = start + chunkSize + (i < remainder? 1: 0);
                    threads.emplace_back(CalcSomeLACErrPro_MRED, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMin), std::ref(pd), start, end, nFrame, isSign, metrType, lacType, RealCom);
                    start = end;
                }
                for (auto& thread: threads)
                    thread.join();
            }
        }
    }
    else {  // nOutput != 1
        vector<vector<bigInt> > YAcc(nFrame, vector<bigInt>(nOutput, 0));
        vector<vector<bigInt> > Y(nFrame, vector<bigInt>(nOutput, 0));
        vector<bigInt> oldErr(nFrame, 0);
        for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
            YAcc[iPatt] = accSmlt.GetOutpPro_complex(iPatt, isSign, nOutput);
            Y[iPatt] = appSmlt.GetOutpPro_complex(iPatt, isSign, nOutput);
            assert(Y[iPatt].size() == nOutput);
        }
        if (metrType == METR_TYPE::MSE) {
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                bigInt diff = 0;
                bigInt square = 0;
                for (ll k = 0; k < nOutput; k++){
                    Y[iPatt][k] += RealCom[k];
                    diff = Y[iPatt][k] - YAcc[iPatt][k];
                    square += diff * diff;
                }
                oldErr[iPatt] = square;
            }
        }
        else if (metrType == METR_TYPE::ER) {         
            for (ll iPatt = 0; iPatt < nFrame; ++iPatt) {
                bool flag = false;
                for (ll k = 0; k < nOutput; k++) {
                    if (Y[iPatt][k] != YAcc[iPatt][k]) {
                        flag = true;
                        break;
                    }
                }
                if (flag)
                    oldErr[iPatt] = 1;
                else
                    oldErr[iPatt] = 0;
            }
        }
        cout << "calculating LAC errors" << endl;
        {
            ll lacNum = lacMan.GetLacNum();
            ll realThread = min(nThread, lacNum);
            assert(realThread > 0);
            cout << "real thread number: " << realThread << endl;
            ll chunkSize = lacNum / realThread;
            ll remainder = lacNum % realThread;
            timer::progress_display pd(lacMan.GetLacNum());
            bigInt runMin = uppBound + 1;
            const ll smallFrame = 10048;
            assert(smallFrame <= nFrame);
            vector<thread> threads;
            ll start = 0;
            for (ll i = 0; i < realThread; ++i) {
                ll end = start + chunkSize + (i < remainder? 1: 0);
                threads.emplace_back(CalcSomeLACErr_complex, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMin), std::ref(pd), start, end, smallFrame, isSign, nOutput, metrType, lacType, 0, 0, RealCom);
                start = end;
            }
            for (auto& thread: threads)
                thread.join();
        }
        // collect promising LACs
        const double perc_large = 0.1;
        const double perc_small = 0.0001;
        if (lacType == LAC_TYPE::SASIMI && 0){
            lacMan.Filt(perc_small);
            cout << "Filt = " <<  perc_small << endl;
        }
        else{
            lacMan.Filt(perc_large);
            cout << "Filt = " <<  perc_large << endl;
        }
        {
            ll lacNum = lacMan.GetLacNum();
            ll realThread = min(nThread, lacNum);
            assert(realThread > 0);
            cout << "real thread number: " << realThread << endl;
            ll chunkSize = lacNum / realThread;
            ll remainder = lacNum % realThread;
            timer::progress_display pd(lacMan.GetLacNum());
            bigInt runMin = uppBound + 1;
            vector<thread> threads;
            ll start = 0;
            for (ll i = 0; i < realThread; ++i) {
                ll end = start + chunkSize + (i < remainder? 1: 0);
                threads.emplace_back(CalcSomeLACErr_complex, std::ref(appSmlt), std::ref(lacMan), std::ref(bdPo2Nodes), std::ref(YAcc), std::ref(oldErr), std::ref(runMin), std::ref(pd), start, end, nFrame, isSign, nOutput, metrType, lacType, 0, 0, RealCom);
                start = end;
            }
            for (auto& thread: threads)
                thread.join();
        }
    }
}

// calculate disjoint cut for LOs and collect cut networks
void VECBEEManPro::BuildCutNtks(NetMan & net) {
    djCuts2.resize(LOs2.size());
    cutNtks2.resize(LOs2.size());
    ll i = 0;
    for (const auto& vLO : LOs2) {
        Abc_NtkIncrementTravId(net.GetNet());
        FindDisjointCutofMultNodes(net, vLO, djCuts2[i]);
        Abc_Obj_t * pNode = nullptr;
        int j = 0;
        Abc_NtkForEachObj(net.GetNet(), pNode, j) {     // PI + node
            if (Abc_ObjIsPo(pNode))
                continue;
            if (Abc_NodeIsTravIdCurrent(pNode))
                cutNtks2[i].emplace_back(pNode);
        }
        Abc_NtkForEachPo(net.GetNet(), pNode, j) {
            if (Abc_NodeIsTravIdCurrent(pNode))
                cutNtks2[i].emplace_back(pNode);
        }
        ++i;
    }
    cout << "finish finding accurate disjoint cut for SubCkts2!" << endl;

    djCuts3.resize(LOs3.size());
    cutNtks3.resize(LOs3.size());
    i = 0;
    for (const auto& vLO : LOs3) {
        Abc_NtkIncrementTravId(net.GetNet());
        FindDisjointCutofMultNodes(net, vLO, djCuts3[i]);
        Abc_Obj_t * pNode = nullptr;
        int j = 0;
        Abc_NtkForEachObj(net.GetNet(), pNode, j) {     // PI + node
            if (Abc_ObjIsPo(pNode))
                continue;
            if (Abc_NodeIsTravIdCurrent(pNode))
                cutNtks3[i].emplace_back(pNode);
        }
        Abc_NtkForEachPo(net.GetNet(), pNode, j) {
            if (Abc_NodeIsTravIdCurrent(pNode))
                cutNtks3[i].emplace_back(pNode);
        }
        ++i;
    }
    cout << "finish finding accurate disjoint cut for SubCkts3!" << endl;
}

void VECBEEManPro::FindDisjointCutofMultNodes(NetMan & net, vector<ll> objIds, list <Abc_Obj_t *> & djCut)
{
    djCut.clear();
    for (ll objId : objIds) {
        Abc_Obj_t * pObj = net.GetObj(objId);
        assert(pObj != nullptr);
        ExpandCut(pObj, djCut);
    }

    Abc_Obj_t * pObjExpd = nullptr;
    while ((pObjExpd = ExpandWhich(djCut)) != nullptr)
        ExpandCut(pObjExpd, djCut);
}

Abc_Obj_t * VECBEEManPro::ExpandWhich(list <Abc_Obj_t *> & disjCut) {
    for (auto ppAbcObj1 = disjCut.begin(); ppAbcObj1 != disjCut.end(); ++ppAbcObj1) {
        auto ppAbcObj2 = ppAbcObj1;
        for (++ppAbcObj2; ppAbcObj2 != disjCut.end(); ++ppAbcObj2) {
            #ifdef DEBUG
            assert(poMarks[(*ppAbcObj1)->Id].size() == poMarks[(*ppAbcObj2)->Id].size());
            assert((*ppAbcObj1)->Id != (*ppAbcObj2)->Id);
            assert(topoIds[(*ppAbcObj1)->Id] != topoIds[(*ppAbcObj2)->Id]);
            #endif
            auto isJoint = poMarks[(*ppAbcObj1)->Id] & poMarks[(*ppAbcObj2)->Id];
            if (isJoint.any()) {
                abc::Abc_Obj_t * pRet = nullptr;
                if (topoIds[(*ppAbcObj1)->Id] < topoIds[(*ppAbcObj2)->Id]) {
                    pRet = *ppAbcObj1;
                    disjCut.erase(ppAbcObj1);
                }
                else {
                    pRet = *ppAbcObj2;
                    disjCut.erase(ppAbcObj2);
                }
                return pRet;
            }
        }
    }
    return nullptr;
}

void VECBEEManPro::CalcBdCut2Node(Simulator & appSmlt, std::vector <int> & vLO2Relation) {
    bdCut2Nodes11.resize(LOs2.size());
    bdCut2Nodes10.resize(LOs2.size());
    vLO2Relation.resize(LOs2.size());
    ll i = 0;
    for (const auto& vLO : LOs2) {
        assert(vLO.size() == 2);
        vLO2Relation[i] = appSmlt.CalcLocBd2(vLO[0], vLO[1], djCuts2[i], cutNtks2[i], bdCut2Nodes11[i], bdCut2Nodes10[i]);
        ++i;
    }
    cout << "finish CalcBdCut2Node for LO2!" << endl;

    bdCut2Nodes101.resize(LOs3.size());
    bdCut2Nodes110.resize(LOs3.size());
    bdCut2Nodes011.resize(LOs3.size());
    bdCut2Nodes111.resize(LOs3.size());
    i = 0;
    for (const auto& vLO : LOs3) {
        assert(vLO.size() == 3);
        appSmlt.CalcLocBd3(vLO[0], vLO[1], vLO[2], djCuts3[i], cutNtks3[i], bdCut2Nodes101[i], bdCut2Nodes110[i], bdCut2Nodes011[i], bdCut2Nodes111[i]);
        // if (i == 0) {
        //     cout << "LOs: " << vLO[0] << ", " << vLO[1] << ", " << vLO[2] << endl;
        //     cout << "cut: ";
        //     for (const auto & pCut : djCuts2[i])
        //         cout << pCut->Id << ", ";
        //     cout << endl;
        // }
        ++i;
    }
    cout << "finish CalcBdCut2Node for LO3!" << endl;
}

void VECBEEManPro::CalcPoBd2(ll PoId, ll LoId, boost::dynamic_bitset <ull> & bdPo2Node11, boost::dynamic_bitset <ull> & bdPo2Node10) {
    bool f10isEmpty = false;
    if (bdCut2Nodes10[LoId].empty())
        f10isEmpty = true;
    
    ll i = 0;
    for (auto pCut: djCuts2[LoId]) {   
        bdPo2Node11 |= bdPo2NodesRef[PoId][pCut->Id] & bdCut2Nodes11[LoId][i]; 
        if (!f10isEmpty)
            bdPo2Node10 |= bdPo2NodesRef[PoId][pCut->Id] & bdCut2Nodes10[LoId][i];             
        ++i;
    }
}

void VECBEEManPro::CalcPoBd3(ll PoId, ll LoId, boost::dynamic_bitset <ull> & bdPo2Node101, boost::dynamic_bitset <ull> & bdPo2Node110, boost::dynamic_bitset <ull> & bdPo2Node011, boost::dynamic_bitset <ull> & bdPo2Node111) {
    ll i = 0;
    for (auto pCut: djCuts3[LoId]) {  
        if (bdCut2Nodes101[LoId][i].size() != 100032) {
            cout << "PoId = " << PoId << ", LoId = " << LoId << ", i = " << i << endl;
            cout << "pCut->Id = " << pCut->Id << endl;
            cout << "bdCut2Nodes101[LoId][i].size() = " << bdCut2Nodes101[LoId][i].size() << endl;
            cout << "bdCut2Nodes110[LoId][i].size() = " << bdCut2Nodes110[LoId][i].size() << endl;
            cout << "bdCut2Nodes011[LoId][i].size() = " << bdCut2Nodes011[LoId][i].size() << endl;
            cout << "bdCut2Nodes111[LoId][i].size() = " << bdCut2Nodes111[LoId][i].size() << endl;
        } 
        bdPo2Node101 |= bdPo2NodesRef[PoId][pCut->Id] & bdCut2Nodes101[LoId][i]; 
        bdPo2Node110 |= bdPo2NodesRef[PoId][pCut->Id] & bdCut2Nodes110[LoId][i]; 
        bdPo2Node011 |= bdPo2NodesRef[PoId][pCut->Id] & bdCut2Nodes011[LoId][i]; 
        bdPo2Node111 |= bdPo2NodesRef[PoId][pCut->Id] & bdCut2Nodes111[LoId][i];             
        ++i;
    }
}


double GetErrFromPoValue(const std::vector < boost::dynamic_bitset <ull> > & accDat, const std::vector < boost::dynamic_bitset <ull> > & appDat, bool isSign, ll nOutput, METR_TYPE metrType, bool fDebug, double errUppBound) {
    if (metrType == METR_TYPE::ER)
        return GetErrRate(accDat, appDat, isSign, nOutput, errUppBound);
    else if (metrType == METR_TYPE::MED)
        return GetMeanErrDist(accDat, appDat, isSign, nOutput, errUppBound);
    else if (metrType == METR_TYPE::MSE)
        return GetMeanSquareErr(accDat, appDat, isSign, nOutput, fDebug, errUppBound);
    else if (metrType == METR_TYPE::MRED)
        return GetMeanRelErrDist(accDat, appDat, isSign, nOutput, errUppBound);
    else if (metrType == METR_TYPE::MHD)
        return GetMeanHamDist(accDat, appDat, isSign, nOutput, errUppBound);
    else
        assert(0);
}


double GetErrRate(const std::vector < boost::dynamic_bitset <ull> > & accDat, const std::vector < boost::dynamic_bitset <ull> > & appDat, bool isSign, ll nOutput, double errUppBound) {
    ll nPo = accDat.size();
    assert(appDat[0].size() <= accDat[0].size());
    ll nFrame = appDat[0].size();
    ll errPattNum = 0;
    ll stopBound = errUppBound * nFrame;
    if (nOutput == 1) {
        for (ll i = 0; i < nFrame; ++i) {
            bool fError = false;
            for (ll o = 0; o < nPo; ++o) {
                if (accDat[o][i] != appDat[o][i]) {
                    fError = true;
                    break;
                }
            }
            if (fError) {
                ++errPattNum;
                if (errPattNum > stopBound) {   // early termination!
                    errPattNum *= 2; 
                    break;
                }
            }
        }
        return double(bigFlt(errPattNum) / bigFlt(nFrame));
    }
    else
        assert(0);
}

double GetMeanErrDist(const std::vector < boost::dynamic_bitset <ull> > & accDat, const std::vector < boost::dynamic_bitset <ull> > & appDat, bool isSign, ll nOutput, double errUppBound) {
    ll nPo = accDat.size();
    assert(appDat[0].size() <= accDat[0].size());
    ll nFrame = appDat[0].size();
    bigInt sed = 0;
    bigInt stopBound = static_cast<bigInt> (errUppBound * nFrame * 1.5);
    if (nOutput == 1) {
        for (ll i = 0; i < nFrame; ++i) {
            boost::dynamic_bitset <ull> accPo(nPo);
            boost::dynamic_bitset <ull> appPo(nPo);
            for (ll o = 0; o < nPo; ++o) {
                accPo[o] = accDat[o][i];
                appPo[o] = appDat[o][i];
            }
            bigInt accOut = GetDecOut(accPo, nPo, isSign);
            bigInt appOut = GetDecOut(appPo, nPo, isSign);
            sed += abs(accOut - appOut);
            if (sed > stopBound) {      // early termination!
                break;
            }
        }
        return double(bigFlt(sed) / bigFlt(nFrame));
    }
    else
        assert(0);
}

double GetMeanSquareErr(const std::vector < boost::dynamic_bitset <ull> > & accDat, const std::vector < boost::dynamic_bitset <ull> > & appDat, bool isSign, ll nOutput, bool fDebug, double errUppBound) {
    ll nPo = accDat.size();
    assert(appDat[0].size() <= accDat[0].size());
    ll nFrame = appDat[0].size();
    bigInt sse = 0;
    bigInt stopBound = static_cast<bigInt> (errUppBound * nFrame * 1.5);
    if (nOutput == 1) {
        for (ll i = 0; i < nFrame; ++i) {
            boost::dynamic_bitset <ull> accPo(nPo);
            boost::dynamic_bitset <ull> appPo(nPo);
            for (ll o = 0; o < nPo; ++o) {
                accPo[o] = accDat[o][i];
                appPo[o] = appDat[o][i];
            }
            bigInt accOut = GetDecOut(accPo, nPo, isSign);
            bigInt appOut = GetDecOut(appPo, nPo, isSign);
            if (fDebug) {
                cout << appOut << ", ";
                if ((i + 1) % 10 == 0)
                    cout << endl;
            }
            sse += (accOut - appOut) * (accOut - appOut);
            if (sse > stopBound) {      // early termination!
                break;
            }
        }
        return double(bigFlt(sse) / bigFlt(nFrame));
    }
    else
        assert(0);
}

double GetMeanRelErrDist(const std::vector < boost::dynamic_bitset <ull> > & accDat, const std::vector < boost::dynamic_bitset <ull> > & appDat, bool isSign, ll nOutput, double errUppBound) {
    ll nPo = accDat.size();
    assert(appDat[0].size() <= accDat[0].size());
    ll nFrame = appDat[0].size();
    bigFlt sum = 0;
    bigFlt stopBound = errUppBound * nFrame * 1.5;
    if (nOutput == 1) {
        for (ll i = 0; i < nFrame; ++i) {
            boost::dynamic_bitset <ull> accPo(nPo);
            boost::dynamic_bitset <ull> appPo(nPo);
            for (ll o = 0; o < nPo; ++o) {
                accPo[o] = accDat[o][i];
                appPo[o] = appDat[o][i];
            }
            bigInt accOut = GetDecOut(accPo, nPo, isSign);
            bigInt appOut = GetDecOut(appPo, nPo, isSign);
            if (accOut != 0)
                sum += abs(1 - static_cast <bigFlt>(appOut)/static_cast <bigFlt>(accOut));
            else
                sum += abs(static_cast <bigFlt>(appOut));
            
            if (sum > stopBound) {  // early termination!
                break;
            }
        }
        return double(sum / bigFlt(nFrame));
    }
    else
        assert(0);
}

double GetMeanHamDist(const std::vector < boost::dynamic_bitset <ull> > & accDat, const std::vector < boost::dynamic_bitset <ull> > & appDat, bool isSign, ll nOutput, double errUppBound) {
    ll nPo = accDat.size();
    assert(appDat[0].size() <= accDat[0].size());
    ll nFrame = appDat[0].size();
    bigInt hd = 0;
    bigInt stopBound = static_cast<bigInt> (errUppBound * nFrame);
    bool fBreak = false;
    if (nOutput == 1) {
        for (ll i = 0; i < nFrame; ++i) {
            for (ll o = 0; o < nPo; ++o) {
                if (accDat[o][i] != appDat[o][i]) {
                    ++hd;
                    if (hd > stopBound) {       // early termination!
                        hd *= 2;
                        fBreak = true;
                        break;
                    }
                }
            }
            if (fBreak)
                break;
        }
        return double(bigFlt(hd) / bigFlt(nFrame));
    }
    else
        assert(0);
}


boost::dynamic_bitset<ull> bigIntToBin(const bigInt& val, ll nPo, bool isSign) {
    // if (isSign) {
    //     bool potentialSignBitIsSet = false;
    //     if (nPo > 0) {
    //          potentialSignBitIsSet = boost::multiprecision::bit_test(val, static_cast<unsigned>(nPo - 1));
    //     }
    //     else
    //         assert(0);
    // }

    // boost::dynamic_bitset<ull> bits(static_cast<size_t>(nPo)); 

    // for (ll i = 0; i < nPo; ++i) {
    //     // bits[0] is LSB
    //     if (boost::multiprecision::bit_test(val, static_cast<unsigned>(i))) {
    //         bits[static_cast<size_t>(i)] = 1; // or bits.set(static_cast<size_t>(i), true);
    //     }
    // }
    // return bits;

    assert(nPo > 0);
    boost::dynamic_bitset<ull> bits(static_cast<size_t>(nPo));

    bigInt v = val;

    if (isSign && val < 0) {
        // Two's complement for negative values (nPo-bit PO space)
        bigInt maxVal = bigInt(1) << nPo;  // 2^nPo
        v = maxVal + val;  // two's complement: 2^nPo + val when val < 0
    }

    for (ll i = 0; i < nPo; ++i) {
        if (boost::multiprecision::bit_test(v, static_cast<unsigned>(i))) {
            bits.set(static_cast<size_t>(i));
        }
    }

    return bits;
}

void VECBEEMan::CleanForNewLacCalc() {
    disjCuts.clear();
    cutNtks.clear();
    poMarks.clear();
    topoIds.clear();
    lacType = LAC_TYPE::SASIMI;
}