#include "lac.h"


using namespace std;
using namespace abc;


void LACMan::GenConstLACs(NetMan & net, std::vector <ll> & targIds) {
    pLacs.reserve(targIds.size() * 2);
    // net.GetLev();
    // ll maxlev = 0;
    // for (auto const & targId: targIds) {
    //     if (net.GetObjLev(targId) > maxlev)
    //         maxlev = net.GetObjLev(targId);
    // }
    for (auto targId: targIds) {
        if (!net.IsNode(targId))
            continue;
        if (net.IsConst(targId))
            continue;
        // if (net.GetObjLev(targId) > 0.5 * maxlev)
        //         continue;
        pLacs.emplace_back(make_shared <ConstLAC> (numeric_limits <double>::max(), targId, true));
        pLacs.emplace_back(make_shared <ConstLAC> (numeric_limits <double>::max(), targId, false));
    }
}

void LACMan::GenConstLACs_ForUpdate(NetMan & net) {
    pLacs.clear();
    Abc_Obj_t * pObj = nullptr;
    int i;
    Abc_NtkForEachNode(net.GetNet(), pObj, i){
    // for (auto targId: targIds) {
        auto targId = net.GetId(pObj);
        if (!net.IsConst(targId))
            continue;
        // cout << net.GetName(targId) << endl;
        if (net.IsConst0(targId)){
            pLacs.emplace_back(make_shared <ConstLAC> (numeric_limits <double>::max(), targId, false));
            pLacs.emplace_back(make_shared <ConstLAC> (numeric_limits <double>::max(), targId, true));
            // cout << "add const1 LAC" << endl;
        }
        if (net.IsConst1(targId)){
            pLacs.emplace_back(make_shared <ConstLAC> (numeric_limits <double>::max(), targId, true));
            pLacs.emplace_back(make_shared <ConstLAC> (numeric_limits <double>::max(), targId, false));
            // cout << "add const0 LAC" << endl;
        }
    }
}


void LACMan::GenSasimiLACsAll(NetMan & net, std::vector <ll> & targIds) {
    pLacs.clear();
    if (net.GetNetType() == NET_TYPE::SOP) {
        net.GetLev();
        for (auto const & targId: targIds) {
            if (!net.IsNode(targId))
                continue;
            if (net.IsConst(targId))
                continue;
            for (ll subId = 0; subId < net.GetIdMaxPlus1(); ++subId) {
                if (!net.IsObj(subId))
                    continue;
                if (net.IsObjPo(subId))
                    continue;
                if (net.IsConst(subId))
                    pLacs.emplace_back(make_shared <SasimiLAC> (numeric_limits <double>::max(), targId, subId, false));
                else if (targId != subId && net.GetObjLev(subId) < net.GetObjLev(targId)) {
                    if (net.GetObjLev(subId) < net.GetObjLev(targId) - 1) {
                        pLacs.emplace_back(make_shared <SasimiLAC> (numeric_limits <double>::max(), targId, subId, false));
                        pLacs.emplace_back(make_shared <SasimiLAC> (numeric_limits <double>::max(), targId, subId, true));
                    }
                    else
                        pLacs.emplace_back(make_shared <SasimiLAC> (numeric_limits <double>::max(), targId, subId, false));
                }
            }
        }
    }
    else if (net.GetNetType() == NET_TYPE::GATE) {
        auto pLibScl = static_cast <SC_Lib *> (net.GetAbcFame()->pLibScl);
        const double invDelay = (pLibScl == nullptr)? net.GetInvDelay(): 40; // delay of CKND0BWP7T30P140HVT is 33.3ps
        net.GetDelay();
        for (auto const & targId: targIds) {
            if (!net.IsNode(targId))
                continue;
            if (net.IsConst(targId))
                continue;
            for (ll subId = 0; subId < net.GetIdMaxPlus1(); ++subId) {
                if (!net.IsObj(subId))
                    continue;
                if (net.IsObjPo(subId))
                    continue;
                if (net.IsConst(subId))
                    pLacs.emplace_back(make_shared <SasimiLAC> (numeric_limits <double>::max(), targId, subId, false));
                else if (targId != subId && DoubleLessEqual(net.GetArrTime(subId), net.GetArrTime(targId), DELAY_TOL)) {
                    if (DoubleLessEqual(net.GetArrTime(subId), net.GetArrTime(targId) - invDelay, DELAY_TOL)) {
                        pLacs.emplace_back(make_shared <SasimiLAC> (numeric_limits <double>::max(), targId, subId, false));
                        if (!net.IsInv(targId))
                            pLacs.emplace_back(make_shared <SasimiLAC> (numeric_limits <double>::max(), targId, subId, true));
                    }
                    else
                        pLacs.emplace_back(make_shared <SasimiLAC> (numeric_limits <double>::max(), targId, subId, false));
                }
            }
        }
    }
    else
        assert(0);
    // for (auto const & lac: pLacs) {
    //     auto sasimi = dynamic_pointer_cast <SasimiLAC> (lac);
    //     cout << sasimi->GetTargId() << "\t" << sasimi->GetSubstId() << "\t" << sasimi->GetErr() << endl;
    // }
}


void LACMan::GenSasimiLACsNew(NetMan & net, std::vector <ll> & targIds) {
    pLacs.clear();
    if (net.GetNetType() == NET_TYPE::SOP || net.GetNetType() == NET_TYPE::GATE) {
        net.GetLev();
        ll maxlev = 0;
        for (auto const & targId: targIds) {
            if (net.GetObjLev(targId) > maxlev)
                maxlev = net.GetObjLev(targId);
        }
        cout << "maxlev = " << maxlev << endl;
        for (auto const & targId: targIds) {
            if (!net.IsNode(targId))
                continue;
            if (net.IsConst(targId))
                continue;
            // If inputs are not uniformly distributed, the following two lines may not apply (consider removing)
            // if (net.GetObjLev(targId) > 0.5 * maxlev)
            //     continue;
            for (ll subId = 0; subId < net.GetIdMaxPlus1(); ++subId) {
                if (!net.IsObj(subId))
                    continue;
                if (net.IsObjPo(subId))
                    continue;
                if (!net.IsNode(subId))
                    continue;
                if (targId == subId)
                    continue;
                if (net.GetObjLev(subId) <= net.GetObjLev(targId))
                    pLacs.emplace_back(make_shared <SasimiLAC> (numeric_limits <double>::max(), targId, subId, false));
                if (net.GetObjLev(subId) <= net.GetObjLev(targId) - 1 && !net.IsInv(targId))
                    pLacs.emplace_back(make_shared <SasimiLAC> (numeric_limits <double>::max(), targId, subId, true));
            }
        }
    }
    else
        assert(0);
    // for (auto const & lac: pLacs) {
    //     auto sasimi = dynamic_pointer_cast <SasimiLAC> (lac);
    //     cout << sasimi->GetTargId() << "\t" << sasimi->GetSubstId() << "\t" << sasimi->GetErr() << endl;
    // }
}


void LACMan::GenRacLACsNew(NetMan & net, std::vector <ll> & targIds, unsigned seed) {
    #ifdef DEBUG
    assert(net.GetNetType() == NET_TYPE::SOP);
    #endif
    const ll nGrowthLevel = 0;
    const ll nCandLimit = 32;
    const ll appFrame = 16;

    // simulate
    Simulator smlt(net, seed, appFrame);
    smlt.InpUnif();
    smlt.Sim();

    // initialize
    pLacs.clear();
    net.GetLev();
    Abc_NtkStartReverseLevels(net.GetNet(), nGrowthLevel);
    
    // generate LACs
    // cout << "generating LACs" << endl;
    // boost::timer::progress_display pd(targIds.size());
    for (auto targId: targIds) {
        if (!net.IsNode(targId) || net.IsConst(targId)) {
            // ++pd;
            continue;
        }
        auto pPivot = net.GetObj(targId);
        // compute divisors
        auto divs = GetDivs(pPivot, Abc_ObjRequiredLevel(pPivot) - 1);
        // cout << pPivot << ":";
        // PrintVect(divs, "\n");
        // enumerate resubstitution
        ll nFanin = net.GetFaninNum(pPivot);
        ll targLev = net.GetObjLev(targId);
        ll countFocus = 0;
        for (ll i = 0; i < nFanin; ++i) {
            ll ithFanin = net.GetFaninId(pPivot, i);
            if (net.GetObjLev(ithFanin) == targLev - 1)
                ++countFocus;
        }
        if (countFocus == 1) {
            ll cnt = 0;
            for (ll i = 0; i < nFanin; ++i) {
                // skip small levels
                ll ithFanin = net.GetFaninId(pPivot, i);
                if (net.GetObjLev(ithFanin) != targLev - 1)
                    continue;
                // init temp divisors
                vector <ll> faninIds;
                faninIds.reserve(10);
                for (int j = 0; j < nFanin; ++j) {
                    if (i != j)
                        faninIds.emplace_back(net.GetFaninId(pPivot, j));
                }
                // try removing the i-th fanin
                if (nFanin > 1) {
                    string func = BuildFuncWithEspresso(smlt, pPivot, faninIds);
                    if (func != string("")) {
                        // cout << pPivot << " approx \n"; PrintVect(faninIds, "\n"); cout << func << endl;
                        pLacs.emplace_back(make_shared <RacLAC> (numeric_limits <double>::max(), targId, faninIds, func));
                    }
                }
                // try replacing the i-th fanin with another divisor
                if (nFanin >= 1) {
                    set <ll> faninIdSet(faninIds.begin(), faninIds.end());
                    faninIds.emplace_back(-1);
                    for (const auto & div: divs) {
                        if (div == ithFanin || faninIdSet.count(div) || net.GetObjLev(div) >= targLev - 1)
                            continue;
                        faninIds.back() = div;
                        string func = BuildFuncWithEspresso(smlt, pPivot, faninIds);
                        // cout << "try " << pPivot << " with ";
                        // PrintVect(faninIds, "\n");
                        if (func != string("")) {
                            // cout << pPivot << " approx \n"; PrintVect(faninIds, "\n"); cout << func << endl;
                            pLacs.emplace_back(make_shared <RacLAC> (numeric_limits <double>::max(), targId, faninIds, func));
                            if (++cnt > nCandLimit)
                                break;
                        }
                    }
                }
            }
        }
        // add constant
        pLacs.emplace_back(make_shared <RacLAC> (numeric_limits <double>::max(), targId, vector <ll> (), string(" 0\n")));
        pLacs.emplace_back(make_shared <RacLAC> (numeric_limits <double>::max(), targId, vector <ll> (), string(" 1\n")));
        // ++pd;
    }

    Abc_NtkStopReverseLevels(net.GetNet());
}


void LACMan::GenSubWireLACs(NetMan & net, std::vector <ll> & targIds) {
    pLacs.clear();
    if (net.GetNetType() == NET_TYPE::SOP || net.GetNetType() == NET_TYPE::GATE) {
        net.GetLev();
        for (auto const & targId: targIds) {
            if (net.IsObjPi(targId))
                continue;
            if (net.IsConst(targId))
                continue;
            for (ll subId = 0; subId < net.GetIdMaxPlus1(); ++subId) {
                if (!net.IsObj(subId))
                    continue;
                if (net.IsObjPo(subId))
                    continue;
                if (!net.IsNode(subId))
                    continue;
                for (ll iFanin = 0; iFanin < net.GetFaninNum(targId); ++iFanin) {
                    ll targWire = net.GetFaninId(targId, iFanin);
                    if (targWire == subId)
                        continue;
                    if (net.GetObjLev(subId) < net.GetObjLev(targWire))
                        pLacs.emplace_back(make_shared <SubWireLAC> (numeric_limits <double>::max(), targId, subId, iFanin, false));
                    if (!net.IsConst(subId) && !net.IsInv(targWire) && net.GetObjLev(subId) < net.GetObjLev(targWire) - 1)
                        pLacs.emplace_back(make_shared <SubWireLAC> (numeric_limits <double>::max(), targId, subId, iFanin, true));
                }
            }
        }
    }
    else
        assert(0);
}


void LACMan::Filt(double perc) {
    // calculate size
    assert(pLacs.size());
    assert(perc >= 0.0 && perc <= 1.0);
    ll newSize = floor(pLacs.size() * perc);
    newSize = max(newSize, 1ll);
    newSize = min(newSize, maxHighAccNumb);

    // Find the position of the n-th element in the sorted sequence
    std::nth_element(pLacs.begin(), pLacs.begin() + newSize, pLacs.end(), 
        [](const auto & A, const auto & B) {return A->GetErrPro() < B->GetErrPro();}
    );

    // sort the first n elements in descending order
    pLacs.resize(newSize);
    sort(pLacs.begin(), pLacs.end(),
        [](const auto & A, const auto & B) {return A->GetTargId() < B->GetTargId();}
    );
    // for (const auto & pLac: pLacs)
    //     dynamic_pointer_cast <ConstLAC> (pLac)->Print();
}

void LACMan::FiltPro(double perc, NetMan & net) {
    // calculate size
    assert(pLacs.size());
    assert(perc >= 0.0 && perc <= 1.0);
    ll newSize = floor(pLacs.size() * perc);
    newSize = max(newSize, 1ll);
    newSize = min(newSize, maxHighAccNumb);

    // Find the position of the n-th element in the sorted sequence
    std::nth_element(pLacs.begin(), pLacs.begin() + newSize, pLacs.end(), 
        [](const auto & A, const auto & B) {return A->GetErrPro() < B->GetErrPro();}
    );

    // deal with subNodes
    for (ll i = newSize; i < pLacs.size(); ++i) {
        auto pLac = GetLac(i);
        ll targId = pLac->GetTargId();
        Abc_Obj_t * pNode = net.GetObj(targId);
        pNode->fMarkC = 0;      // Clear marks for LAC target nodes; other LACs on this node may still be low-error (see LacPerSubNode)
    }

    // sort the first n elements in descending order
    pLacs.resize(newSize);
    sort(pLacs.begin(), pLacs.end(),
        [](const auto & A, const auto & B) {return A->GetTargId() < B->GetTargId();}
    );
    // for (const auto & pLac: pLacs)
    //     dynamic_pointer_cast <ConstLAC> (pLac)->Print();
}


shared_ptr <LAC> LACMan::GetBestLac() const {
    #ifdef DEBUG
    assert(pLacs.size());
    #endif
    double bestErr = numeric_limits <double>::max();
    shared_ptr <LAC> pBestLac = pLacs[0];
    // ll count = 0;
    for (const auto & pLac: pLacs) {
        // cout << "LAC[" << count++ << "] with Err = "  << pLac->GetErr() << ", current best = " << bestErr << endl;
        if (bestErr > pLac->GetErr()) {
            bestErr = pLac->GetErr();
            pBestLac = pLac;
        }
    }
    return pBestLac;
}

vector < shared_ptr <LAC> > LACMan::GetMultBestLac() const {
    #ifdef DEBUG
    assert(pLacs.size());
    #endif
    vector <bigInt> bestErrs;
    vector < shared_ptr <LAC> > pBestLacs;
    ll LacSize = 2;
    // ll count = 0;
    auto pBestLac = pLacs[0];
    bigInt bestErr = pBestLac->GetErrPro();
    for (auto i = 0; i < LacSize; ++i){
        pBestLacs.emplace_back(pBestLac);
        bestErrs.emplace_back(bestErr); 
    }
    for (ll i = 1; i < pLacs.size(); ++i) {
        auto pLac = pLacs[i];
        bigInt err = pLac->GetErrPro();
        // cout << "LAC[" << i << "] with Err = "  << err << ", current best1 = " << bestErrs[0] << ", current best2 = " << bestErrs[1] << endl;
        if (bestErrs[1] > err) {
            if(bestErrs[0] > err){
                bestErrs[1] = bestErrs[0];
                pBestLacs[1] = pBestLacs[0];
                bestErrs[0] = err;
                pBestLacs[0] = pLac;
            }
            else{
                bestErrs[1] = err;
                pBestLacs[1] = pLac;
            }
        }
    }
    return pBestLacs;
}


shared_ptr <LAC> LACMan::GetBestLacPro() const {
    assert(pLacs.size());
    auto pBestLac = pLacs[0];
    bigInt bestErr = pBestLac->GetErrPro();
    for (ll i = 1; i < pLacs.size(); ++i) {
        auto pLac = pLacs[i];
        bigInt err = pLac->GetErrPro();
        // cout << "LAC[" << i << "] with Err = "  << err << ", current best = " << bestErr << endl;
        if (bestErr > err) {
            bestErr = err;
            pBestLac = pLac;
        }
    }
    return pBestLac;
}

// vector <shared_ptr <LAC>> LACMan::GetNegErrLac() const {
//     vector <shared_ptr<LAC>> NegErrLacs;
//     for (ll i = 0; i < pLacs.size(); ++i) {
//         auto pLac = pLacs[i];
//         bigInt err = pLac->GetErrPro();
//         auto Id = pLac->GetTargId();
//         if (err < 0)
//             NegErrLacs.emplace_back(make_shared <LAC> (err, Id));
//     }
//     return NegErrLacs;
// }


extern "C" {
    Vec_Ptr_t * Abc_MfsWinMarkTfi(Abc_Obj_t * pNode);
    void Abc_MfsWinSweepLeafTfo_rec(Abc_Obj_t * pObj, int nLevelLimit);
}
vector <ll> LACMan::GetDivs(Abc_Obj_t * pNode, ll nLevDivMax) {
    const ll nWinMax = 300;
    const ll nFanoutsMax = 30;
    vector <ll> divs;
    Vec_Ptr_t * vCone, * vDivs;
    Abc_Obj_t * pObj, * pFanout, * pFanin;
    int k, f, m;
    int nDivsPlus = 0, nTrueSupp;

    // mark the TFI with the current trav ID
    Abc_NtkIncrementTravId( pNode->pNtk );
    vCone = Abc_MfsWinMarkTfi( pNode );

    // count the number of PIs
    nTrueSupp = 0;
    Vec_PtrForEachEntry( Abc_Obj_t *, vCone, pObj, k )
        nTrueSupp += Abc_ObjIsCi(pObj);
//    printf( "%d(%d) ", Vec_PtrSize(p->vSupp), m );

    // mark with the current trav ID those nodes that should not be divisors:
    // (1) the node and its TFO
    // (2) the MFFC of the node
    // (3) the node's fanins (these are treated as a special case)
    Abc_NtkIncrementTravId( pNode->pNtk );
    Abc_MfsWinSweepLeafTfo_rec( pNode, nLevDivMax );
//    Abc_MfsWinVisitMffc( pNode );
    Abc_ObjForEachFanin( pNode, pObj, k )
        Abc_NodeSetTravIdCurrent( pObj );

    // at this point the nodes are marked with two trav IDs:
    // nodes to be collected as divisors are marked with previous trav ID
    // nodes to be avoided as divisors are marked with current trav ID

    // start collecting the divisors
    vDivs = Vec_PtrAlloc( nWinMax );
    Vec_PtrForEachEntry( Abc_Obj_t *, vCone, pObj, k )
    {
        if ( !Abc_NodeIsTravIdPrevious(pObj) )
            continue;
        if ( (int)pObj->Level > nLevDivMax )
            continue;
        Vec_PtrPush( vDivs, pObj );
        if ( Vec_PtrSize(vDivs) >= nWinMax )
            break;
    }
    Vec_PtrFree( vCone );

    // explore the fanouts of already collected divisors
    if ( Vec_PtrSize(vDivs) < nWinMax )
    Vec_PtrForEachEntry( Abc_Obj_t *, vDivs, pObj, k )
    {
        // consider fanouts of this node
        Abc_ObjForEachFanout( pObj, pFanout, f )
        {
            // stop if there are too many fanouts
            if ( nFanoutsMax && f > nFanoutsMax )
                break;
            // skip nodes that are already added
            if ( Abc_NodeIsTravIdPrevious(pFanout) )
                continue;
            // skip nodes in the TFO or in the MFFC of node
            if ( Abc_NodeIsTravIdCurrent(pFanout) )
                continue;
            // skip COs
            if ( !Abc_ObjIsNode(pFanout) )
                continue;
            // skip nodes with large level
            if ( (int)pFanout->Level > nLevDivMax )
                continue;
            // skip nodes whose fanins are not divisors  -- here we skip more than we need to skip!!! (revise later)  August 7, 2009
            Abc_ObjForEachFanin( pFanout, pFanin, m )
                if ( !Abc_NodeIsTravIdPrevious(pFanin) )
                    break;
            if ( m < Abc_ObjFaninNum(pFanout) )
                continue;
            // make sure this divisor in not among the nodes
//            Vec_PtrForEachEntry( Abc_Obj_t *, p->vNodes, pFanin, m )
//                assert( pFanout != pFanin );
            // add the node to the divisors
            Vec_PtrPush( vDivs, pFanout );
            // Vec_PtrPushUnique( p->vNodes, pFanout );
            Abc_NodeSetTravIdPrevious( pFanout );
            nDivsPlus++;
            if ( Vec_PtrSize(vDivs) >= nWinMax )
                break;
        }
        if ( Vec_PtrSize(vDivs) >= nWinMax )
            break;
    }

    // sort the divisors by level in the increasing order
    Vec_PtrSort( vDivs, (int (*)(const void *, const void *))Abc_NodeCompareLevelsIncrease );

    // add the fanins of the node
    Abc_ObjForEachFanin( pNode, pFanin, k )
        Vec_PtrPush( vDivs, pFanin );
    
    divs.reserve(Vec_PtrSize(vDivs));
    Vec_PtrForEachEntry(Abc_Obj_t *, vDivs, pObj, k)
        divs.emplace_back(pObj->Id);

    // clean up
    Vec_PtrFree(vDivs);

    return divs;
}


string LACMan::BuildFuncWithEspresso(Simulator & smlt, Abc_Obj_t * pPivot, const vector <ll> & faninIds) {
    Abc_Ntk_t * pAppNtk = smlt.GetNet();
    ll nFrame = smlt.GetFrameNumb();

    assert(pAppNtk == pPivot->pNtk);
    assert(faninIds.size() >= 1);

    // check the existence of resubstitution and build truth table
    typedef unordered_map <string, bool> table_t;
    table_t truthTable;
    for (int i = 0; i < nFrame; ++i) {
        string minterm("");
        // Vec_PtrForEachEntry(Abc_Obj_t *, vFanins, pFanin, k)
        for (const auto & faninId: faninIds)
            minterm += (*smlt.GetDat(faninId))[i]? '1': '0';
        bool val = (*smlt.GetDat(pPivot->Id))[i];
        table_t::const_iterator got = truthTable.find(minterm);
        if (got == truthTable.end())
            truthTable[minterm] = val;
        else {
            if (got->second != val)
                return string("");
        }
    }
    // for (const auto & [key, value]: truthTable)
    //     cout << key << "," << value << endl;

    // construct function with espresso
    int nVars = faninIds.size();
    pPLA PLA = new_PLA();
    PLA->pla_type = FR_type;

    assert(cube.fullset == nullptr);
    cube.num_binary_vars = nVars;
    cube.num_vars = cube.num_binary_vars + 1;
    cube.part_size = ALLOC(int, cube.num_vars);

    assert(cube.fullset == nullptr);
    assert(cube.part_size != nullptr);
    cube.part_size[cube.num_vars-1] = 1;
    cube_setup();
    PLA_labels(PLA);

    if (PLA->F == nullptr) {
        PLA->F = new_cover(10);
        PLA->D = new_cover(10);
        PLA->R = new_cover(10);
    }

    pcube cf = cube.temp[0];
    for (table_t::const_iterator iter = truthTable.begin(); iter != truthTable.end(); ++iter) {
        set_clear(cf, cube.size);
        const string & minterm = iter->first;
        for (int i = 0; i < nVars; ++i) {
            if (minterm[i] == '0')
                set_insert(cf, 2*i);
            else if (minterm[i] == '1')
                set_insert(cf, 2*i+1);
            else
                assert(0);
        }
        set_insert(cf,2*nVars);
        if (iter->second)
            PLA->F = sf_addset(PLA->F, cf);
        else
            PLA->R = sf_addset(PLA->R, cf);
    }

    pcover X;
    free_cover(PLA->D);
    X = d1merge(sf_join(PLA->F, PLA->R), cube.num_vars - 1);
    PLA->D = complement(cube1list(X));
    free_cover(X);

    PLA->F = espresso(PLA->F, PLA->D, PLA->R);

    // convert to sop
    string func("");
    pcube last, c;
    foreach_set(PLA->F, last, c) {
        for (int var = 0; var < cube.num_binary_vars; var++) {
            char item = ("?01-" [GETINPUT(c, var)]);
            assert(item != '?');
            func += item;
        }
        assert(cube.num_binary_vars == cube.num_vars - 1);
        assert(cube.output != -1);
        int llast = cube.last_part[cube.output];
        assert(cube.first_part[cube.output] == llast);
        assert("01" [is_in_set(c, llast) != 0] == '1');
        func += " 1\n";
    }

    // clean up
    free_PLA(PLA);
    FREE(cube.part_size);
    setdown_cube();
    sf_cleanup();
    sm_cleanup();

    return func;
}


void LACMan::GenCandLacs() {
    ll maxId = -1;
    for (ll i = 0; i < GetLacNum(); ++i)
        maxId = max(maxId, GetLac(i)->GetTargId());
    #ifdef DEBUG
    assert(maxId != -1);
    #endif
    candLacs.resize(maxId + 1, nullptr);

    vector <double> minErrs(maxId + 1, numeric_limits <double>::max());
    for (ll i = 0; i < GetLacNum(); ++i) {
        auto pLac =  GetLac(i);
        auto targId = pLac->GetTargId();
        #ifdef DEBUG
        assert(pLac->GetErr() >= 0.0);
        #endif
        if (pLac->GetErr() < minErrs[targId]) {
            candLacs[targId] = pLac;
            minErrs[targId] = pLac->GetErr();
        }
    }
}


void LACMan::GenCandLacs(const vector <ll> & critGraph) {
    ll maxId = -1;
    for (ll i = 0; i < GetLacNum(); ++i)
        maxId = max(maxId, GetLac(i)->GetTargId());
    #ifdef DEBUG
    assert(maxId != -1);
    #endif
    candLacs.resize(maxId + 1, nullptr);

    vector <double> minErrs(maxId + 1, numeric_limits <double>::max());
    set <ll> S(critGraph.begin(), critGraph.end());
    for (ll i = 0; i < GetLacNum(); ++i) {
        auto pLac =  GetLac(i);
        auto targId = pLac->GetTargId();
        if (S.count(targId) == 0)
            continue;
        #ifdef DEBUG
        assert(pLac->GetErr() >= 0.0);
        #endif
        if (pLac->GetErr() < minErrs[targId]) {
            candLacs[targId] = pLac;
            minErrs[targId] = pLac->GetErr();
        }
    }
}

void LACMan::PrintLACsErr() {
    for (ll i = 0; i < GetLacNum(); ++i) {
        auto pLac = GetLac(i);
        cout << "i = " << i << ", err = " << pLac->GetErrPro() << endl;
    }
}

void SNGMan::CreateNode(Abc_Obj_t * pNode) {
    nNodes++;
    std::shared_ptr<SNGNode> newNode = std::make_shared<SNGNode>(pNode, nNodes);
    vSubNodes.push_back(newNode);
}

void SNGMan::CreateNode(std::shared_ptr<SNGNode> pSNGNode) {
    vSubNodes.push_back(pSNGNode);
}

std::shared_ptr <SNGNode> SNGMan::GetNode(ll id) const {
    if (id > 0 && id <= vSubNodes.size()) {
        return vSubNodes[id - 1];  // SNGId starts from 1, but vector start from 0
    }
    cout << "! error id = " << id << ", vSubNodes.size() = " << vSubNodes.size() << endl;
    return nullptr;
}

bool SNGNode::FindIdInTotalFanouts(ll id) {
    if (find(vTotalFanouts.begin(), vTotalFanouts.end(), id) != vTotalFanouts.end())
        return true;
    else
        return false;
}

bool SNGNode::FindIdInTotalFanins(ll id) {
    if (find(vTotalFanins.begin(), vTotalFanins.end(), id) != vTotalFanins.end())
        return true;
    else
        return false;
}

void SNGMan::ClearGraph() {
    vSubNodes.clear();
    vLos.clear();
    vLis.clear();
    errInc.clear();
    nNodes = 0;
}

void SNGMan::UpdateErrInc(std::unordered_map <ll, std::shared_ptr <LAC>> & LacPerSubNode) {
    errInc.resize(nNodes);
    ll i = 0;
    for (const auto& node : vSubNodes) {
        ll netId = node->GetNetId();
        assert(i == node->GetSNGId() - 1);
        auto it = LacPerSubNode.find(netId);
        if (it != LacPerSubNode.end()) {
            errInc[i] = it->second->GetErr();
        }
        else {
            cout << "netId = " << netId << endl;
            assert(0);
        }
        ++i;
    }
}

double SNGMan::GenNearDisCut(NetMan & net, vector <ll> & nearDisCut, double errUppBound, double backErr, METR_TYPE metrType, double jointTh, ll & nRoundLowGain) {
    vector <vector <ll>> cuts;
    vector <double> cutsErrSum;
    vector <double> cutsJointScore;
    vector <ll> cutsMffcSum;
    assert(nNodes == vSubNodes.size());
    cuts.resize(nNodes);
    cutsErrSum.resize(nNodes);
    cutsJointScore.resize(nNodes);
    cutsMffcSum.resize(nNodes);
    ll i = 0;
    // CleanfMarkA();
    // CleanfMarkB();
    
    cout << "jointness threshold = " << jointTh << endl;

    cout << "Start generating near-disjoint cuts!" << endl;
    double jointMinRef = 1.0;
    double jointRefSum = 0.0;
    double boundRatio = 0.9;
    for (const auto& node : vSubNodes) { 
        double jointLocalRef = 1.0;
        net.CleanTravIds();
        cuts[i].push_back(node->GetNetId());
        ll nodeNum = 1;
        assert(i == node->GetSNGId() - 1);
        Abc_Obj_t * pNode = node->GetpNode();

        // for cut node excluding
        // MarkTFO_rec(node->GetSNGId());
        // MarkTFI_rec(node->GetSNGId());
        node->SetfMarkA();

        // for jointness calculation
        ll TfoSizeSum = 0;
        ll TfoNodeNum = 0;
        Abc_NtkCleanMarkA(net.GetNet());
        net.CalcTFO(pNode, TfoSizeSum, TfoNodeNum, 1);

        double jointScore = 0;
        double errSum = errInc[i];
        double mffcSum = node->GetnMffc();

        bool fContinue = true;
        while (fContinue) {
            ll candNetId = 0;
            ll candSNGId = 0;
            // bool fNoNodes = true;
            double smallestScore = -1;
            double joint = 0;
            ll addnMffc = 0;

            // select a node to add in cut
            for (ll j = 1; j <= nNodes; ++j) {
                if (errSum + errInc[j - 1] + backErr > boundRatio * errUppBound)
                    continue;
                auto pSNGNode = GetNode(j);
                if (pSNGNode->GetfMarkA())   // don't consider nodes that are already added
                    continue;
                Abc_Obj_t * pNode2 = pSNGNode->GetpNode();

                // calculate jointness score
                ll TfoSize = 0;
                ll nAddNum = 0;
                net.CalcTFO(pNode2, TfoSize, nAddNum, 0);
                ll tmpTfoSizeSum = TfoSizeSum + TfoSize;
                ll tmpTfoNodeNum = TfoNodeNum + nAddNum;

                double tmpJointScore = (double(nodeNum + 1) / double(nodeNum)) * (1 - double(tmpTfoNodeNum) / double(tmpTfoSizeSum));
                if ((tmpJointScore > jointTh) && (tmpJointScore < jointLocalRef))
                    jointLocalRef = tmpJointScore;
                

                double normErr = CalcNormErr2(metrType, backErr, errInc[j - 1]);
                if (normErr < 0) {
                    cout << "errInc[j - 1] = " << errInc[j - 1] << ", backErr = " << backErr << endl;
                    assert(normErr >= 0);
                }
                double tmpScore = normErr / double(pSNGNode->GetnMffc()) / (jointTh - tmpJointScore);
                bool fUpdate = false;
                if (smallestScore == -1)    // still initial value
                    fUpdate = true;
                else if (smallestScore < 0 && tmpScore >= 0)
                    fUpdate = true;
                else if ((smallestScore > 0) && (tmpScore > 0) && (tmpScore < smallestScore))
                    fUpdate = true;
                else if ((smallestScore == 0) && (tmpScore == 0)) {     // for normErr == 0 case
                    if (double(pSNGNode->GetnMffc()) * (jointTh - tmpJointScore) > double(addnMffc) * (jointTh - joint))
                        fUpdate = true;
                }               
                if ((smallestScore != -1) && (tmpJointScore > jointTh))    // total check: for normErr == 0 case, still do not allow exceeding jointTh (except for initialization)
                    fUpdate = false;
                
                if (fUpdate) {    // select the best single node
                    smallestScore = tmpScore;
                    joint = tmpJointScore;
                    candNetId = pSNGNode->GetNetId();
                    candSNGId = j;
                    addnMffc = pSNGNode->GetnMffc();
                }
            }

            // update (add a node to current cut)
            if ((joint <= jointTh) && (candNetId != 0)) {
                cuts[i].push_back(candNetId);
                ++nodeNum;
                // MarkTFO_rec(candSNGId); 
                // MarkTFI_rec(candSNGId); 
                auto pSNGNode = GetNode(candSNGId);
                pSNGNode->SetfMarkA();
                Abc_Obj_t * pNetNode = net.GetObj(candNetId); 
                ll var1, var2;
                net.CalcTFO(pNetNode, var1, var2, 1);
                TfoSizeSum += var1;
                TfoNodeNum += var2;
                errSum += errInc[candSNGId - 1];
                mffcSum += addnMffc;
                jointScore = joint;

                jointLocalRef = 1.0;
            }
            else {
                fContinue = false;
                jointMinRef = min(jointMinRef, jointLocalRef);
                jointRefSum += jointLocalRef;
            }
        }

        cutsJointScore[i] = jointScore;
        cutsErrSum[i] = errSum;
        cutsMffcSum[i] = mffcSum;

        ++i;
        CleanfMarkA();
        // CleanfMarkB();
    }

    double jointAvgRef = jointRefSum / double(nNodes);
    cout << "jointMinRef = " << jointMinRef << ", jointAvgRef = " << jointAvgRef << endl;
    vector <double> cutsMffcArea;
    cutsMffcArea.resize(nNodes);

    cout << "Start selecting the subsets of cuts!" << endl;
    // for each cut, select the best subset
    vector <vector <ll>> subcuts = cuts;
    vector <double> subcutsErrSum, subcutsAreaSum, subcutsScore;
    subcutsErrSum.resize(nNodes);
    subcutsAreaSum.resize(nNodes);
    subcutsScore.resize(nNodes);
    
    for (ll i = 0; i < nNodes; ++i) {   // for cuts[i]
        net.CleanTravIds();
        double currReArea = net.CalcMultiNodesMffcArea(cuts[i]);
        double currErrInc = cutsErrSum[i];
        bool isCont = true;
        double newSmallestRatio;
        while (isCont) {    // continue to choose to remove one node from the set
            if (currReArea == 0) {
                cout << "i = " << i << " (size = " << subcuts[i].size() << "): ";
                for (ll j = 0; j < subcuts[i].size(); ++j) {
                    cout << cuts[i][j] << ", ";
                }
                cout << endl;
                net.PrintPro(1, 1, 0);
                assert(currReArea != 0);
            }
            double ratio = CalcNormErr2(metrType, backErr, currErrInc) / currReArea;
            newSmallestRatio = ratio;

            if (subcuts[i].size() == 1)
                break;

            // if (subcuts[i].size() == 2) {
            //     if (subcuts[i][0] == 324 && subcuts[i][1] == 487) {
            //         ;
            //     }
            // }

            ll bestId = 0;  // pNtk's ID
            ll bestSngId = 0;
            double bestArea = 0;
            ll bestCutItemId = -1;
            for (ll j = 0; j < subcuts[i].size(); ++j) {
                ll nodeId = subcuts[i][j];
                Abc_Obj_t * pNode = net.GetObj(nodeId);
                Abc_NtkIncrementTravId(net.GetNet());
                Abc_MfsWinMarkTfi(pNode);
                for (ll k = 0; k < subcuts[i].size(); ++k) {
                    if (k == j)
                        continue;
                    Abc_Obj_t * pNode2 = net.GetObj(subcuts[i][k]);
                    if (Abc_NodeIsTravIdCurrent(pNode2))
                        Abc_NodeSetTravIdPrevious(pNode2);
                }

                Abc_Obj_t * pMffc;
                ll k;
                double area = 0;
                Abc_NtkForEachNode(net.GetNet(), pMffc, k) {
                    if (pMffc->fMarkD && Abc_NodeIsTravIdCurrent(pMffc)) {
                        area += net.GetNodeArea(pMffc);
                    }                   
                }   

                // If the remaining node is a MFFC of the removed node, (currReArea - area) will be 0. In this case, we cannot choose this node to remove. In addition, norm error must be non-negative for number comparison.
                // if (currReArea - area == 0)
                //     continue;
                
                double tmpRatio = CalcNormErr2(metrType, backErr, currErrInc - errInc[pNode->sngId - 1]) / (currReArea - area);
                if (tmpRatio < newSmallestRatio) {
                    newSmallestRatio = tmpRatio;
                    bestId = nodeId;
                    bestSngId = pNode->sngId;
                    bestArea = area;
                    bestCutItemId = j;
                }
            }
            if (newSmallestRatio == ratio)
                isCont = false;     // terminate
            else {
                subcuts[i].erase(subcuts[i].begin() + bestCutItemId);
                currErrInc -= errInc[bestSngId - 1];
                currReArea -= bestArea;

                // if (fDebug)
                //     cout << "errInc[bestId] = " << errInc[bestId] << ", currErrInc = " << currErrInc << endl;

                Abc_NtkIncrementTravId(net.GetNet());
                Abc_MfsWinMarkTfi(net.GetObj(bestId));
                for (ll k = 0; k < subcuts[i].size(); ++k) {
                    Abc_Obj_t * pNode2 = net.GetObj(subcuts[i][k]);
                    if (Abc_NodeIsTravIdCurrent(pNode2))
                        Abc_NodeSetTravIdPrevious(pNode2);
                }

                Abc_Obj_t * pMffc;
                ll k;
                Abc_NtkForEachNode(net.GetNet(), pMffc, k) {
                    if (pMffc->fMarkD && Abc_NodeIsTravIdCurrent(pMffc)) {
                        pMffc->fMarkD = 0;
                    }                   
                }
            }
        }
        subcutsErrSum[i] = currErrInc;
        subcutsAreaSum[i] = currReArea;
        subcutsScore[i] = newSmallestRatio;

        // if (fDebug)
        //     cout << endl;
    }

    cout << "start selecting the best subset!" << endl;
    // select the best subset
    ll bestId = 0;
    for (ll i = 1; i < nNodes; ++i) {
        if (subcutsScore[i] < subcutsScore[bestId])
            bestId = i;
    }
    nearDisCut = subcuts[bestId];

    cout << "cut size = " << nearDisCut.size() << ": ";
    for (ll i = 0; i < nearDisCut.size(); ++i) {
        cout << nearDisCut[i];
        Abc_Obj_t * pNode = net.GetObj(nearDisCut[i]);
        if (pNode->fMarkF)
            cout << "(F) ";
        else
            cout << " ";
    }
    cout << endl;
    cout << "errInc = " << subcutsErrSum[bestId] << "(estimated whole error = " << subcutsErrSum[bestId] + backErr << "), reArea = " << subcutsAreaSum[bestId] << endl;
    cout << "(the original cut size = " << cuts[bestId].size() << ")" << endl << endl;
    
    if (subcutsAreaSum[bestId] < 5.0 && nearDisCut.size() < 5)
        ++nRoundLowGain;
    else
        nRoundLowGain = 0;
    if (nearDisCut.size() == 1)
        ++nRoundLowGain;  // +2, must increase jointTh

    if (jointAvgRef < 0.45)
        return (jointAvgRef + 0.05);
    else
        // return jointAvgRef;
        return 0.5;
}

extern "C" {
    int Abc_MfsWinVisitMffc(Abc_Obj_t * pNode);
}
ll SNGMan::FindBestSingleLAC(NetMan & net, double errUppBound, double backErr, METR_TYPE metrType) {
    ll i = 0;
    net.CleanTravIds();
    double smallestRatio = 0.0;
    ll bestNetId = 0;
    double bestErrInc = 0;
    double bestReArea = 0;
    for (const auto& node : vSubNodes) {
        assert(i == node->GetSNGId() - 1);
        Abc_Obj_t * pNode = node->GetpNode();
        double normErr = CalcNormErr2(metrType, backErr, errInc[i]);

        Abc_NtkIncrementTravId(net.GetNet());
        // Abc_MfsWinMarkTfi(pNode);
        Abc_MfsWinVisitMffc(pNode);
        Abc_Obj_t * pMffc;
        ll k;
        double mffcArea = 0;
        Abc_NtkForEachNode(net.GetNet(), pMffc, k) {
            if (Abc_NodeIsTravIdCurrent(pMffc)) {
                mffcArea += net.GetNodeArea(pMffc);
            }                   
        }

        if (bestNetId == 0) {
            // smallestRatio = normErr / mffcArea;
            smallestRatio = normErr;
            bestNetId = pNode->Id;
            bestErrInc = errInc[i];
            bestReArea = mffcArea;
        }
        else {
            // double tmpRatio = normErr / mffcArea;
            double tmpRatio = normErr;
            if (tmpRatio < smallestRatio) {
                smallestRatio = tmpRatio;
                bestNetId = pNode->Id;
                bestErrInc = errInc[i];
                bestReArea = mffcArea;
            }
        }
        ++i;
    }
    cout << "bestNetId = " << bestNetId << ", errIncrease = " << bestErrInc << "(whole error = " << bestErrInc + backErr << "), reArea = " << bestReArea << endl;
    auto pNode = net.GetObj(bestNetId);
    cout << "pNode->markValue = " << pNode->markValue << endl;
    return bestNetId;
}


// void SNGMan::MarkTFI(ll id) {
//     vector <ll> tranFanins;
//     tranFanins.push_back(id);
//     while (!tranFanins.empty()) {
//         vector <ll> tmpTranFanins;
//         for (ll i = 0; i < tranFanins.size(); ++i) {
//             auto node1 = GetNode(tranFanins[i]);
//             node1->SetfMarkA();
//             for (ll j = 0; j < node1->GetTotalFaninNum(); ++j) {
//                 ll faninId = node1->GetFanin(j);
//                 auto node2 = GetNode(faninId);
//                 node2->SetfMarkA();
//                 if (node2->GetType() != SNG_OBJ_TYPE::SNG_LI)
//                     tmpTranFanins.push_back(faninId);
//             }
//         }
//         tranFanins = tmpTranFanins;
//     }
// }

// void SNGMan::MarkTFO(ll id) {
//     vector <ll> tranFanouts;
//     tranFanouts.push_back(id);
//     ++currTravId;
//     while (!tranFanouts.empty()) {
//         vector <ll> tmpTranFanouts;
//         for (ll i = 0; i < tranFanouts.size(); ++i) {
//             auto node1 = GetNode(tranFanouts[i]);
//             node1->SetTravId(currTravId);
//             node1->SetfMarkA();
//             for (ll j = 0; j < node1->GetTotalFanoutNum(); ++j) {
//                 ll fanoutId = node1->GetFanout(j);
//                 auto node2 = GetNode(fanoutId);
//                 node2->SetfMarkA();
//                 if ((node2->GetType() != SNG_OBJ_TYPE::SNG_LO) || (node2->IsTravIdCurrent(currTravId)))
//                     tmpTranFanouts.push_back(fanoutId);
//             }
//         }
//         tranFanouts = tmpTranFanouts;
//     }
// }

void SNGMan::MarkTFO_rec(ll id) {   // use fMarkA
    auto node = GetNode(id);
    node->SetfMarkA();
    if (node->GetType() == SNG_OBJ_TYPE::SNG_LO)
        return;
    for (ll i = 0; i < node->GetTotalFanoutNum(); ++i) {
        auto fanout = GetNode(node->GetFanout(i)); 
        if (!fanout->GetfMarkA())
            MarkTFO_rec(node->GetFanout(i));
    }
}

void SNGMan::MarkTFI_rec(ll id) {   // use fMarkB
    auto node = GetNode(id);
    node->SetfMarkB();
    if (node->GetType() == SNG_OBJ_TYPE::SNG_LI)
        return;
    for (ll i = 0; i < node->GetTotalFaninNum(); ++i) {
        auto fanin = GetNode(node->GetFanin(i)); 
        if (!fanin->GetfMarkB())
            MarkTFI_rec(node->GetFanin(i));
    }
}

void SNGMan::CleanfMarkA() {
    for (const auto& node : vSubNodes) {
        node->ResetfMarkA();
    }
}

void SNGMan::CleanfMarkB() {
    for (const auto& node : vSubNodes) {
        node->ResetfMarkB();
    }
}

void SNGMan::Clear() {
    vSubNodes.clear();
    vLos.clear();
    vLis.clear();
    errInc.clear();
    nNodes = 0;
    currTravId = 0;
}

void SNGNode::SetnMffc(NetMan & net) {
    nMffc = net.GetNodeMffcSize(pNode);
}

void SNGMan::UpdatenMffc(NetMan & net) {
    for (const auto& node : vSubNodes) {
        node->SetnMffc(net);
    }
}

double CalcNormErr(METR_TYPE metrType, double backErr, double errInc) {
    if (metrType == METR_TYPE::MSE) {
        if (errInc + backErr <= 0)
            return -std::numeric_limits<double>::infinity();
        return log2(sqrt(errInc + backErr));
    }
    else if (metrType == METR_TYPE::MED) {
        if (errInc + backErr <= 0)
            return -std::numeric_limits<double>::infinity();
        return log2(errInc + backErr);
    }
    else if (metrType == METR_TYPE::ER)
        return (errInc + backErr);
    else
        assert(0);
}

double CalcNormErr2(METR_TYPE metrType, double backErr, double errInc) {
    if (metrType == METR_TYPE::MSE) {
        if (errInc + backErr < 0)
            return -std::numeric_limits<double>::infinity();
        return log2(1 + sqrt(errInc + backErr));
    }
    else if (metrType == METR_TYPE::MED) {
        if (errInc + backErr <= 0)
            return -std::numeric_limits<double>::infinity();
        return log2(1 + errInc + backErr);
    }
    else if (metrType == METR_TYPE::ER)
        return (errInc + backErr);
    else
        assert(0);
}

std::pair<double, ll> SynthFunction(ll tableValue, ll nVars) {
    // build truth table
    typedef unordered_map <string, bool> table_t;
    table_t truthTable;
    if (nVars == 4) {
        std::bitset<16> table(tableValue);  // for 4-input function
        for (ll i = 0; i < 16; ++i) {   // for 4-input function
            std::bitset<4> pattern(i);
            truthTable[pattern.to_string()] = table[i];     // i: start from the right
        }
    }
    else if (nVars == 3) {
        std::bitset<8> table(tableValue); 
        for (ll i = 0; i < 8; ++i) {   
            std::bitset<3> pattern(i);
            truthTable[pattern.to_string()] = table[i];     
        }
    }
    else if (nVars == 2) {
        std::bitset<4> table(tableValue); 
        for (ll i = 0; i < 4; ++i) {   
            std::bitset<2> pattern(i);
            truthTable[pattern.to_string()] = table[i];   
        }
    }
    else
        assert(0);

    // cout << "truth table = " << tableValue << endl;

    // construct function with espresso
    pPLA PLA = new_PLA();
    PLA->pla_type = FR_type;

    assert(cube.fullset == nullptr);
    cube.num_binary_vars = nVars;
    cube.num_vars = cube.num_binary_vars + 1;
    cube.part_size = ALLOC(int, cube.num_vars);

    assert(cube.fullset == nullptr);
    assert(cube.part_size != nullptr);
    cube.part_size[cube.num_vars-1] = 1;
    cube_setup();
    PLA_labels(PLA);

    if (PLA->F == nullptr) {
        PLA->F = new_cover(10);
        PLA->D = new_cover(10);
        PLA->R = new_cover(10);
    }

    pcube cf = cube.temp[0];
    for (table_t::const_iterator iter = truthTable.begin(); iter != truthTable.end(); ++iter) {
        set_clear(cf, cube.size);
        const string & minterm = iter->first;
        for (int i = 0; i < nVars; ++i) {
            if (minterm[i] == '0')
                set_insert(cf, 2*i);
            else if (minterm[i] == '1')
                set_insert(cf, 2*i+1);
            else
                assert(0);
        }
        set_insert(cf,2*nVars);
        if (iter->second)
            PLA->F = sf_addset(PLA->F, cf);
        else
            PLA->R = sf_addset(PLA->R, cf);
    }

    pcover X;
    free_cover(PLA->D);
    X = d1merge(sf_join(PLA->F, PLA->R), cube.num_vars - 1);
    PLA->D = complement(cube1list(X));
    free_cover(X);

    PLA->F = espresso(PLA->F, PLA->D, PLA->R);

    // convert to sop
    string func("");
    pcube last, c;
    foreach_set(PLA->F, last, c) {
        for (int var = 0; var < cube.num_binary_vars; var++) {
            char item = ("?01-" [GETINPUT(c, var)]);
            assert(item != '?');
            func += item;
        }
        assert(cube.num_binary_vars == cube.num_vars - 1);
        assert(cube.output != -1);
        int llast = cube.last_part[cube.output];
        assert(cube.first_part[cube.output] == llast);
        assert("01" [is_in_set(c, llast) != 0] == '1');
        func += " 1\n";
    }

    // clean up
    free_PLA(PLA);
    FREE(cube.part_size);
    setdown_cube();
    sf_cleanup();
    sm_cleanup();

    // cout << "func = ";
    // for (char c : func) {
    //     if (c == '\n')
    //         cout << ", ";
    //     else
    //         cout << c;
    // }
    // cout << endl;

    // Synthesis sop func as a new network
    if (func != "") {
        Abc_Ntk_t * pNtk = Abc_NtkAlloc(ABC_NTK_LOGIC, ABC_FUNC_SOP, 1);
        Abc_Obj_t * pNewNode = Abc_NtkCreateNode(pNtk);
        pNewNode->pData = Abc_SopRegister((Mem_Flex_t *)pNtk->pManFunc, func.c_str());
        Abc_Obj_t ** pFaninNodes = new Abc_Obj_t*[nVars];
        for (int k = 0; k < nVars; ++k) {
            pFaninNodes[k] = Abc_NtkCreatePi(pNtk);
        }
        for (int k = 0; k < nVars; ++k) {
            Abc_ObjAddFanin(pNewNode, pFaninNodes[k]);
        }
        Abc_Obj_t * pOutNode = Abc_NtkCreatePo(pNtk);
        Abc_ObjAddFanin(pOutNode, pNewNode);
        // synthesis
        Abc_Frame_t * pAbc = Abc_FrameGetGlobalFrame();
        Abc_FrameReplaceCurrentNetwork(pAbc, Abc_NtkDup(pNtk));
        // string Command = string("strash; balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance; logic; map;");
        string Command = string("strash; resyn2a; logic; amap;");
        // string Command = string("strash; resyn3; logic; amap;");
        assert(!Cmd_CommandExecute(pAbc, Command.c_str()));
        Abc_NtkDelete(pNtk);
        Abc_Ntk_t * pSynNtk = Abc_NtkDup(Abc_FrameReadNtk(pAbc));
    
        double area = Abc_NtkGetMappedArea(pSynNtk);
        ll nodeNum = Abc_NtkNodeNum(pSynNtk);
        // cout << "area = " << area << ", #node = " << nodeNum << endl << endl;

        // debug
        // Abc_Obj_t * pNode;
        // ll i;
        // cout << "network: " << endl;
        // Abc_NtkForEachNode(pSynNtk, pNode, i) {
        //     cout << "id = " << i << ": " << std::string(abc::Mio_GateReadName(static_cast <abc::Mio_Gate_t *> (pNode->pData))) << endl;
        // }
        // cout << "total area = " << Abc_NtkGetMappedArea(pSynNtk) << endl;

        // clean up
        delete[] pFaninNodes;
        Abc_NtkDelete(pSynNtk);

        return make_pair(area, nodeNum);
    }
    return make_pair(0.0, 0);
}


static std::mutex synth_mutex;

double SynthFunction_MultiOut(std::vector<ll> tableValues, ll nVars) {
    std::lock_guard<std::mutex> lock(synth_mutex);  // serialize this function

    vector <string> sopFuncs;
    // build truth table
    for (ll tableValue: tableValues) {
        typedef unordered_map <string, bool> table_t;
        table_t truthTable;
        if (nVars == 4) {
            std::bitset<16> table(tableValue);  // for 4-input function
            for (ll i = 0; i < 16; ++i) {   // for 4-input function
                std::bitset<4> pattern(i);
                truthTable[pattern.to_string()] = table[i];     // i: start from the right
            }
        }
        else if (nVars == 3) {
            std::bitset<8> table(tableValue); 
            for (ll i = 0; i < 8; ++i) {   
                std::bitset<3> pattern(i);
                truthTable[pattern.to_string()] = table[i];     
            }
        }
        else if (nVars == 2) {
            std::bitset<4> table(tableValue); 
            for (ll i = 0; i < 4; ++i) {   
                std::bitset<2> pattern(i);
                truthTable[pattern.to_string()] = table[i];   
            }
        }
        else
            assert(0);

        // construct function with espresso
        pPLA PLA = new_PLA();
        PLA->pla_type = FR_type;

        assert(cube.fullset == nullptr);
        cube.num_binary_vars = nVars;
        cube.num_vars = cube.num_binary_vars + 1;
        cube.part_size = ALLOC(int, cube.num_vars);

        assert(cube.fullset == nullptr);
        assert(cube.part_size != nullptr);
        cube.part_size[cube.num_vars-1] = 1;
        cube_setup();
        PLA_labels(PLA);

        if (PLA->F == nullptr) {
            PLA->F = new_cover(10);
            PLA->D = new_cover(10);
            PLA->R = new_cover(10);
        }

        pcube cf = cube.temp[0];
        for (table_t::const_iterator iter = truthTable.begin(); iter != truthTable.end(); ++iter) {
            set_clear(cf, cube.size);
            const string & minterm = iter->first;
            for (int i = 0; i < nVars; ++i) {
                if (minterm[i] == '0')
                    set_insert(cf, 2*i);
                else if (minterm[i] == '1')
                    set_insert(cf, 2*i+1);
                else
                    assert(0);
            }
            set_insert(cf,2*nVars);
            if (iter->second)
                PLA->F = sf_addset(PLA->F, cf);
            else
                PLA->R = sf_addset(PLA->R, cf);
        }

        pcover X;
        free_cover(PLA->D);
        X = d1merge(sf_join(PLA->F, PLA->R), cube.num_vars - 1);
        PLA->D = complement(cube1list(X));
        free_cover(X);

        PLA->F = espresso(PLA->F, PLA->D, PLA->R);

        // convert to sop
        string func("");
        pcube last, c;
        foreach_set(PLA->F, last, c) {
            for (int var = 0; var < cube.num_binary_vars; var++) {
                char item = ("?01-" [GETINPUT(c, var)]);
                assert(item != '?');
                func += item;
            }
            assert(cube.num_binary_vars == cube.num_vars - 1);
            assert(cube.output != -1);
            int llast = cube.last_part[cube.output];
            assert(cube.first_part[cube.output] == llast);
            assert("01" [is_in_set(c, llast) != 0] == '1');
            func += " 1\n";
        }

        // clean up
        free_PLA(PLA);
        FREE(cube.part_size);
        setdown_cube();
        sf_cleanup();
        sm_cleanup();

        // assert(func != "");
        sopFuncs.push_back(func);
    }

    // Synthesis sop func as a new network
    Abc_Ntk_t * pNtk = Abc_NtkAlloc(ABC_NTK_LOGIC, ABC_FUNC_SOP, 1);
    Abc_Obj_t ** pFaninNodes = new Abc_Obj_t*[nVars];
    for (int k = 0; k < nVars; ++k) {
        pFaninNodes[k] = Abc_NtkCreatePi(pNtk);
    }
    for (string func : sopFuncs) {
        Abc_Obj_t * pNewNode = Abc_NtkCreateNode(pNtk);
        if (func != "")
            pNewNode->pData = Abc_SopRegister((Mem_Flex_t *)pNtk->pManFunc, func.c_str());
        else 
            pNewNode->pData = Abc_SopCreateConst0((Mem_Flex_t *)pNtk->pManFunc);
        for (int k = 0; k < nVars; ++k) {
            Abc_ObjAddFanin(pNewNode, pFaninNodes[k]);
        }
        Abc_Obj_t * pOutNode = Abc_NtkCreatePo(pNtk);
        Abc_ObjAddFanin(pOutNode, pNewNode);    
    }
    // synthesis
    double area = 0;
    Abc_Frame_t * pAbc = Abc_FrameGetGlobalFrame();
    Abc_FrameReplaceCurrentNetwork(pAbc, Abc_NtkDup(pNtk));
    // string Command = string("strash; balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance; logic; map;");
    string Command = string("strash; resyn2a; logic; amap;");
    // string Command = string("strash; resyn3; logic; amap;");
    assert(!Cmd_CommandExecute(pAbc, Command.c_str()));
    Abc_NtkDelete(pNtk);
    Abc_Ntk_t * pSynNtk = Abc_NtkDup(Abc_FrameReadNtk(pAbc)); 
    area = Abc_NtkGetMappedArea(pSynNtk);

    // clean up
    delete[] pFaninNodes;
    Abc_NtkDelete(pSynNtk);

    return area;
}

set <ll> LACMan::GetScand(ll nCand, METR_TYPE metrType, NetMan & net, ll nFrame) {
    // cout << "pLacs.size() = " << pLacs.size() << ", nCand = " << nCand << endl; 
    // if (nCand <= pLacs.size()) {
    //     if (metrType == METR_TYPE::MRED) {
    //         std::nth_element(pLacs.begin(), pLacs.begin() + nCand, pLacs.end(), 
    //             [](const auto & A, const auto & B) {return A->GetErrBigFlt() < B->GetErrBigFlt();}
    //         );
    //     }
    //     else {
    //         std::nth_element(pLacs.begin(), pLacs.begin() + nCand, pLacs.end(), 
    //             [](const auto & A, const auto & B) {return A->GetErrPro() < B->GetErrPro();}
    //         );
    //     }
    //     pLacs.resize(nCand);
    // }
    // set <ll> Scand;
    // for (const auto & pLac : pLacs) {
    //     Scand.insert(pLac->GetTargId());
    // }
    // return Scand;


    cout << "pLacs.size() = " << pLacs.size() << ", nCand = " << nCand << endl; 
    // const ll nLac = 100 * nCand;     // can be tuned
    // cout << "nLac = " << nLac << endl;
    // if (nLac <= pLacs.size()) {
    //     if (metrType == METR_TYPE::MRED) {
    //         std::nth_element(pLacs.begin(), pLacs.begin() + nLac, pLacs.end(), 
    //             [](const auto & A, const auto & B) {return A->GetErrBigFlt() < B->GetErrBigFlt();}
    //         );
    //     }
    //     else {
    //         std::nth_element(pLacs.begin(), pLacs.begin() + nLac, pLacs.end(), 
    //             [](const auto & A, const auto & B) {return A->GetErrPro() < B->GetErrPro();}
    //         );
    //     }
    //     pLacs.resize(nLac);
    // }

    if (metrType == METR_TYPE::MRED) {
        sort(pLacs.begin(), pLacs.end(),
            [](const auto & A, const auto & B) {return A->GetErrBigFlt() < B->GetErrBigFlt();}
        );
    }
    else {
        sort(pLacs.begin(), pLacs.end(),
            [](const auto & A, const auto & B) {return A->GetErrPro() < B->GetErrPro();}
        );
    }

    set <ll> Scand;
    for (const auto & pLac : pLacs) {
        ll targId = pLac->GetTargId();
        auto it = Scand.find(targId);
        if (it == Scand.end()) {           
            // mark smallest error increase
            Abc_Obj_t * pNode = net.GetObj(targId);
            bigFlt err = -1;
            if (metrType == METR_TYPE::MRED) 
                err = pLac->GetErrBigFlt();
            else
                err = bigFlt(pLac->GetErrPro());
            pNode->errInc = double(err / bigFlt(nFrame));

            // add the node to Scand
            Scand.insert(targId);
            // check size of Scand             
            if (Scand.size() >= nCand)
                break;
        }     
    }
    return Scand;
}

std::shared_ptr <LAC> LACMan::GetLacWithSmallestErr(METR_TYPE metrType) {
    if (metrType == METR_TYPE::MRED) {
        sort(pLacs.begin(), pLacs.end(),
            [](const auto & A, const auto & B) {return A->GetErrBigFlt() < B->GetErrBigFlt();}
        );
    }
    else {
        sort(pLacs.begin(), pLacs.end(),
            [](const auto & A, const auto & B) {return A->GetErrPro() < B->GetErrPro();}
        );
    }
    return pLacs[0];
}

void LACMan::SortLacs(METR_TYPE metrType) {
    if (metrType == METR_TYPE::MRED) {
        sort(pLacs.begin(), pLacs.end(),
            [](const auto & A, const auto & B) {return A->GetErrBigFlt() < B->GetErrBigFlt();}
        );
    }
    else {
        sort(pLacs.begin(), pLacs.end(),
            [](const auto & A, const auto & B) {return A->GetErrPro() < B->GetErrPro();}
        );
    }
}

std::shared_ptr <LAC> LACMan::GetLac(ll i) {
    return pLacs[i];
}

void LACMan::CleanLacs() {
    pLacs.clear();
}

bool LACMan::CheckSasimiLev(NetMan & net) {
    bool fPass = true;
    for (ll lacId = 0; lacId < pLacs.size(); ++lacId) {
        auto pLac = GetLac(lacId);
        ll targId = pLac->GetTargId();
        auto & specLac = *dynamic_pointer_cast <SasimiLAC>(pLac);
        ll subId = specLac.GetSubId();

        ll targLev = net.GetObjLev(targId);
        ll subLev = net.GetObjLev(subId);
        if (subLev > targLev) {
            fPass = false;
            cout << "targId = " << targId << "(lev = " << targLev << "), subId = " << subId << "(lev = " << subLev << ")" << endl;
        }
    }
    if (!fPass)
        net.PrintPro(1, 1, 0);

    return fPass;
}