#include "subCkt.h"

using namespace std;
using namespace abc;
using namespace boost;

void SubCktMan::GenSubCkts(const vector <ll> & Scand) {
    cout << "find 2 nodes as LO" << endl;
    for (ll i = 0; i < Scand.size(); ++i) {
        Abc_Obj_t * pObj1 = net.GetObj(Scand[i]);
        assert(pObj1 != nullptr);
        assert(Abc_ObjIsNode(pObj1));
        assert(!Abc_NodeIsConst(pObj1));
        for (ll j = i + 1; j < Scand.size(); ++j) {
            Abc_Obj_t * pObj2 = net.GetObj(Scand[j]);
            assert(pObj2 != nullptr);
            assert(Abc_ObjIsNode(pObj2));
            assert(!Abc_NodeIsConst(pObj2));

            net.Abc_NtkCleanMarkDE();
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
            vector <ll> vLI;
            double area = 0;
            ll nodeNum = 0;
            while (1) {
                set <ll> LIsNew;
                Abc_Obj_t * pExpandNode;
                ll expandId = 0;
                // select a node to expand forward (the direction to PI)
                for (auto it = LIs.rbegin(); it != LIs.rend(); ++it) {  // topo order (from largest ID to smallest ID)
                    expandId = *it;
                    pExpandNode = net.GetObj(expandId);
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
                        if (LIs.size() <= 5) {
                            ll nMffc = 0;
                            Abc_Obj_t * pObj;
                            Abc_NtkForEachNode(net.GetNet(), pObj, k) {
                                if (pObj->fMarkD) {
                                    ++nMffc;
                                }
                            }
                            if (nMffc >= 2) {
                                vLI.assign(LIs.begin(), LIs.end()); 
                                area = net.CalcMarkArea();
                                nodeNum = nMffc;                              
                            }
                        }
                    }
                }
            }
            if (!vLI.empty()) {
                vector <ll> vLO = {Scand[i], Scand[j]};
                auto p = make_shared<SubCkt>(area, nodeNum, vLI, vLO);
                if (nodeNum > 2)
                    pSubCkts2.push_back(p);
                else
                    pSubCkts2Trivial.push_back(p);
            }
            else {
                assert(nodeNum == 0);
                // assert(area == 0);
            }
        }
    }

    cout << "find 3 nodes as LO" << endl;
    for (ll i = 0; i < Scand.size(); ++i) {
        Abc_Obj_t * pObj1 = net.GetObj(Scand[i]);
        assert(pObj1 != nullptr);
        assert(Abc_ObjIsNode(pObj1));
        assert(!Abc_NodeIsConst(pObj1));
        for (ll j = i + 1; j < Scand.size(); ++j) {
            Abc_Obj_t * pObj2 = net.GetObj(Scand[j]);
            assert(pObj2 != nullptr);
            assert(Abc_ObjIsNode(pObj2));
            assert(!Abc_NodeIsConst(pObj2));
            if (net.IsPathExist2(Scand[i], Scand[j]))
                continue;
            for (ll l = j + 1; l < Scand.size(); ++l) {
                Abc_Obj_t * pObj3 = net.GetObj(Scand[l]);
                assert(pObj3 != nullptr);
                assert(Abc_ObjIsNode(pObj3));
                assert(!Abc_NodeIsConst(pObj3));
                if (net.IsPathExist2(Scand[i], Scand[l]) || net.IsPathExist2(Scand[j], Scand[l]))
                    continue;

                net.Abc_NtkCleanMarkDE();
        
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
                vector <ll> vLI;
                double area = 0;
                ll nodeNum = 0;
                while (1) {
                    set <ll> LIsNew;
                    Abc_Obj_t * pExpandNode;
                    ll expandId = 0;
                    // select a node to expand forward (the direction to PI)
                    for (auto it = LIs.rbegin(); it != LIs.rend(); ++it) {  // topo order (from largest ID to smallest ID)
                        expandId = *it;
                        pExpandNode = net.GetObj(expandId);
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
                            if (LIs.size() <= 5) {
                                ll nMffc = 0;
                                Abc_Obj_t * pObj;
                                Abc_NtkForEachNode(net.GetNet(), pObj, k) {
                                    if (pObj->fMarkD) {
                                        ++nMffc;
                                    }
                                }
                                if (nMffc >= 3) {
                                    vLI.assign(LIs.begin(), LIs.end()); 
                                    area = net.CalcMarkArea();  
                                    nodeNum = nMffc;                            
                                }
                            }
                        }
                    }
                }

                if (!vLI.empty()) {
                    vector <ll> vLO = {Scand[i], Scand[j], Scand[l]};
                    auto p = make_shared<SubCkt>(area, nodeNum, vLI, vLO);
                    if (nodeNum > 3)
                        pSubCkts3.push_back(p);
                    else
                        pSubCkts3Trivial.push_back(p);
                }
                else {
                    assert(area == 0);
                    assert(nodeNum == 0);
                }
            }
        }
    }

    // sort by area
    // sort(pSubCkts2.begin(), pSubCkts2.end(),
    // [](const std::shared_ptr<SubCkt>& a, const std::shared_ptr<SubCkt>& b) {
    //     return a->GetArea() > b->GetArea();
    // });
    // sort(pSubCkts3.begin(), pSubCkts3.end(),
    // [](const std::shared_ptr<SubCkt>& a, const std::shared_ptr<SubCkt>& b) {
    //     return a->GetArea() > b->GetArea();
    // });

    // filter by estimated error
    double perc = 1;  // can be tuned
    ll maxCktNum = 500;    // can be tuned
    cout << "pSubCkts2 filtering: " << pSubCkts2.size() << " -> ";
    for (const auto& pSub : pSubCkts2) {
        double sum = 0;
        double minErr = std::numeric_limits<double>::max();
        for (const auto loId : pSub->GetvLO()) {
            double errInc = net.GetObj(loId)->errInc;
            sum += errInc;
            minErr = min(minErr, errInc);
        }
        pSub->SetErrIncSum(sum);
        pSub->SetMinErr(minErr);
    }
    ll newSize = min(ll(pSubCkts2.size() * perc), maxCktNum);
    std::nth_element(pSubCkts2.begin(), pSubCkts2.begin() + newSize, pSubCkts2.end(),
    [](const std::shared_ptr<SubCkt>& a, const std::shared_ptr<SubCkt>& b) {
        // if (a->GetMinErr() != b->GetMinErr())
        //     return a->GetMinErr() < b->GetMinErr();
        // else if (a->GetErrIncSum() != b->GetErrIncSum())
        if (a->GetErrIncSum() != b->GetErrIncSum())
            return a->GetErrIncSum() < b->GetErrIncSum();
        return a->GetArea() > b->GetArea();
    });
    pSubCkts2.resize(newSize);
    cout << pSubCkts2.size() << endl;

    cout << "pSubCkts3 filtering: " << pSubCkts3.size() << " -> ";
    for (const auto& pSub : pSubCkts3) {
        double sum = 0;
        double minErr = std::numeric_limits<double>::max();
        for (const auto loId : pSub->GetvLO()) {
            double errInc = net.GetObj(loId)->errInc;
            sum += errInc;
            minErr = min(minErr, errInc);
        }
        pSub->SetErrIncSum(sum);
        pSub->SetMinErr(minErr);
    }
    newSize = min(ll(pSubCkts3.size() * perc), maxCktNum);
    std::nth_element(pSubCkts3.begin(), pSubCkts3.begin() + newSize, pSubCkts3.end(),
    [](const std::shared_ptr<SubCkt>& a, const std::shared_ptr<SubCkt>& b) {
        // if (a->GetMinErr() != b->GetMinErr())
        //     return a->GetMinErr() < b->GetMinErr();
        // else if (a->GetErrIncSum() != b->GetErrIncSum())
        if (a->GetErrIncSum() != b->GetErrIncSum())
            return a->GetErrIncSum() < b->GetErrIncSum();
        return a->GetArea() > b->GetArea();
    });
    pSubCkts3.resize(newSize);
    cout << pSubCkts3.size() << endl;

    // for observation
    // for (const auto& pSub : pSubCkts2) {
    //     double sum = 0;
    //     double minErr = std::numeric_limits<double>::max();
    //     for (const auto loId : pSub->GetvLO()) {
    //         double errInc = net.GetObj(loId)->errInc;
    //         sum += errInc;
    //         minErr = min(minErr, errInc);
    //     }
    //     pSub->SetErrIncSum(sum);
    //     pSub->SetMinErr(minErr);
    // }
    // sort(pSubCkts2.begin(), pSubCkts2.end(),
    // [](const std::shared_ptr<SubCkt>& a, const std::shared_ptr<SubCkt>& b) {
    //     if (a->GetErrIncSum() != b->GetErrIncSum())
    //         return a->GetErrIncSum() < b->GetErrIncSum();
    //     return a->GetArea() > b->GetArea();
    // });
    // ll i = 1;
    // for (const auto& pSub : pSubCkts2) {
    //     pSub->SetRank1(i);
    //     ++i;
    // }
    // sort(pSubCkts2.begin(), pSubCkts2.end(),
    // [](const std::shared_ptr<SubCkt>& a, const std::shared_ptr<SubCkt>& b) {
    //     if (a->GetMinErr() != b->GetMinErr())
    //         return a->GetMinErr() < b->GetMinErr();
    //     else if (a->GetErrIncSum() != b->GetErrIncSum())
    //         return a->GetErrIncSum() < b->GetErrIncSum();
    //     return a->GetArea() > b->GetArea();
    // });
    // i = 1;
    // for (const auto& pSub : pSubCkts2) {
    //     pSub->SetRank2(i);
    //     ++i;
    // }

    // for (const auto& pSub : pSubCkts3) {
    //     double sum = 0;
    //     double minErr = std::numeric_limits<double>::max();
    //     for (const auto loId : pSub->GetvLO()) {
    //         double errInc = net.GetObj(loId)->errInc;
    //         sum += errInc;
    //         minErr = min(minErr, errInc);
    //     }
    //     pSub->SetErrIncSum(sum);
    //     pSub->SetMinErr(minErr);
    // }
    // sort(pSubCkts3.begin(), pSubCkts3.end(),
    // [](const std::shared_ptr<SubCkt>& a, const std::shared_ptr<SubCkt>& b) {
    //     if (a->GetErrIncSum() != b->GetErrIncSum())
    //         return a->GetErrIncSum() < b->GetErrIncSum();
    //     return a->GetArea() > b->GetArea();
    // });
    // i = 1;
    // for (const auto& pSub : pSubCkts3) {
    //     pSub->SetRank1(i);
    //     ++i;
    // }
    // sort(pSubCkts3.begin(), pSubCkts3.end(),
    // [](const std::shared_ptr<SubCkt>& a, const std::shared_ptr<SubCkt>& b) {
    //     if (a->GetMinErr() != b->GetMinErr())
    //         return a->GetMinErr() < b->GetMinErr();
    //     else if (a->GetErrIncSum() != b->GetErrIncSum())
    //         return a->GetErrIncSum() < b->GetErrIncSum();
    //     return a->GetArea() > b->GetArea();
    // });
    // i = 1;
    // for (const auto& pSub : pSubCkts3) {
    //     pSub->SetRank2(i);
    //     ++i;
    // }

    // ll size2 = pSubCkts2.size();
    // ll size3 = pSubCkts3.size();
    // cout << "before resizing: #SubCkts2 = " << size2 << ", #SubCkts3 = " << size3 << endl;
    // pSubCkts2.resize(size2 * 0.4);
    // pSubCkts3.resize(size3 * 0.4);
}

void AppRW::PrintOriSubNtk(NetMan & net) {
    net.Abc_NtkCleanMarkDE();
    // fMarkD = 1: is mffc; fMarkE = 1: have been expanded/explored

    set <ll> LIs;
    Abc_Obj_t * pFanin;
    ll k;
    for (const auto & LoId : vLO) {
        auto pLO = net.GetObj(LoId);
        pLO->fMarkD = 1;
        Abc_ObjForEachFanin(pLO, pFanin, k) {
            LIs.insert(pFanin->Id);
        }
    }

    vector <ll> vLI;
    double area = 0;
    ll nodeNum = 0;
    while (1) {
        set <ll> LIsNew;
        Abc_Obj_t * pExpandNode;
        ll expandId = 0;
        // select a node to expand forward (the direction to PI)
        for (auto it = LIs.rbegin(); it != LIs.rend(); ++it) {  // topo order (from largest ID to smallest ID)
            expandId = *it;
            pExpandNode = net.GetObj(expandId);
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
                if (LIs.size() <= 5) {
                    ll nMffc = 0;
                    Abc_Obj_t * pObj;
                    Abc_NtkForEachNode(net.GetNet(), pObj, k) {
                        if (pObj->fMarkD) {
                            ++nMffc;
                        }
                    }
                    if (nMffc >= 2) {
                        vLI.assign(LIs.begin(), LIs.end()); 
                        // area = net.CalcMarkArea();
                        // nodeNum = nMffc;                              
                    }
                }
            }
        }
    }

    Abc_Obj_t * pObj;
    cout << "original sub-circuit: " << endl;
    Abc_NtkForEachNode(net.GetNet(), pObj, k) {
        if (pObj->fMarkD) {
            net.PrintObjPro(k, 1, 1);
        }
    }
    cout << endl;
}

void SubCktMan::Print() {
    cout << "pSubCkts2.size() = " << pSubCkts2.size() << ", pSubCkts3.size() = " << pSubCkts3.size() << endl;

    cout << "#LO = 2: " << endl;
    for (size_t i = 0; i < pSubCkts2.size(); ++i) {
        const auto& sub = pSubCkts2[i];
        cout << "i = " << i << ": ";

        // Print LO
        cout << "  LO: ";
        for (ll lo : sub->GetvLO()) {
            cout << lo << " ";
        }

        // Print LI
        cout << "  LI: ";
        for (ll li : sub->GetvLI()) {
            cout << li << " ";
        }
        // cout << "#LI = " << sub->GetLInum();

        cout << " area = " << sub->GetArea() << ", #nodes = " << sub->GetNodeNum() << endl;
    }

    cout << "#LO = 3: " << endl;
    for (size_t i = 0; i < pSubCkts3.size(); ++i) {
        const auto& sub = pSubCkts3[i];
        cout << "i = " << i << ": ";

        // Print LO
        cout << "  LO: ";
        for (ll lo : sub->GetvLO()) {
            cout << lo << " ";
        }

        // Print LI
        cout << "  LI: ";
        for (ll li : sub->GetvLI()) {
            cout << li << " ";
        }

        cout << " area = " << sub->GetArea() << ", #nodes = " << sub->GetNodeNum() << endl;
    }
}

void SubCktMan::CalcBD(Simulator & appSmlt, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef_, const std::vector<boost::dynamic_bitset<ull>>& poMarks_, const std::vector <ll> & topoIds) {
    vector < vector <ll> > LOs2;
    vector < vector <ll> > LOs3;
    for (const auto& pSub : pSubCkts2) {
        LOs2.push_back(pSub->GetvLO());
    }
    for (const auto& pSub : pSubCkts3) {
        LOs3.push_back(pSub->GetvLO());
    }

    VECBEEManPro vecbeeManPro(bdPo2NodesRef_, poMarks_, ref(LOs2), ref(LOs3), topoIds);
    vecbeeManPro.BuildCutNtks(net);
    vecbeeManPro.CalcBdCut2Node(appSmlt, vLO2Relation);

    // calculate bdPo according to bdCut 
    cout << "calculate bdPo according to bdCut: LO2" << endl;
    ll nPo = appSmlt.GetPoNum();
    bdPo2Nodes11.resize(nPo);
    bdPo2Nodes10.resize(nPo);
    for (ll o = 0; o < nPo; ++o) {
        ll i = 0;
        bdPo2Nodes11[o].resize(LOs2.size());
        bdPo2Nodes10[o].resize(LOs2.size());
        for (const auto& vLO : LOs2) {
            assert(vLO.size() == 2);
            auto & bdPo2Node11 = bdPo2Nodes11[o][i];
            auto & bdPo2Node10 = bdPo2Nodes10[o][i];
            bdPo2Node11.resize(appSmlt.GetFrameNumb(), 0);
            bdPo2Node10.resize(appSmlt.GetFrameNumb(), 0);
            vecbeeManPro.CalcPoBd2(o, i, bdPo2Node11, bdPo2Node10);
            ++i;
        }
    }

    cout << "calculate bdPo according to bdCut: LO3" << endl;
    bdPo2Nodes101.resize(nPo);
    bdPo2Nodes110.resize(nPo);
    bdPo2Nodes011.resize(nPo);
    bdPo2Nodes111.resize(nPo);
    for (ll o = 0; o < nPo; ++o) {
        ll i = 0;
        bdPo2Nodes101[o].resize(LOs3.size());
        bdPo2Nodes110[o].resize(LOs3.size());
        bdPo2Nodes011[o].resize(LOs3.size());
        bdPo2Nodes111[o].resize(LOs3.size());
        for (const auto& vLO : LOs3) {
            assert(vLO.size() == 3);
            auto & bdPo2Node101 = bdPo2Nodes101[o][i];
            auto & bdPo2Node110 = bdPo2Nodes110[o][i];
            auto & bdPo2Node011 = bdPo2Nodes011[o][i];
            auto & bdPo2Node111 = bdPo2Nodes111[o][i];
            bdPo2Node101.resize(appSmlt.GetFrameNumb(), 0);
            bdPo2Node110.resize(appSmlt.GetFrameNumb(), 0);
            bdPo2Node011.resize(appSmlt.GetFrameNumb(), 0);
            bdPo2Node111.resize(appSmlt.GetFrameNumb(), 0);
            vecbeeManPro.CalcPoBd3(o, i, bdPo2Node101, bdPo2Node110, bdPo2Node011, bdPo2Node111);
            ++i;
        }
    }
}

extern "C" {
    Vec_Ptr_t * Abc_MfsWinMarkTfi(Abc_Obj_t * pNode);
}
ll SubCkt::GenDiv(NetMan & net, ll nlevLim) {
    net.GetLev();
    // mark LOs' TFO with fMarkA
    Abc_NtkCleanMarkA(net.GetNet());
    for (const auto & LoId : vLO) {
        auto pLO = net.GetObj(LoId);
        Abc_MfsWinSweepLeafTfo_rec_Pro(pLO);
    }

    // exclude LI which is in the TFO cone of some LO
    vector <ll> vLInew;
    for (const auto & LiId : vLI) {
        if (net.GetObj(LiId)->fMarkA) {
            continue;
        }
        vLInew.push_back(LiId);
    }

    // calculate support PI set
    vector < vector <ll>> PiMarks;
    PiMarks.resize(net.GetIdMaxPlus1());
    for (ll i = 0; i < net.GetIdMaxPlus1(); i++) 
        PiMarks[i].resize(net.GetPiNum());
    Abc_Obj_t * pNode;
    ll i;
    for (ll i = 0; i < net.GetPiNum(); i++) {
        Abc_Obj_t * pPi = net.GetPi(i);
        PiMarks[pPi->Id][i] = 1;
    }
    Abc_NtkForEachNode(net.GetNet(), pNode, i) {
        for (ll j = 0; j < net.GetFaninNum(i); j++) {
            ll faninId = net.GetId(net.GetFanin(i, j));
            if (j == 0) {
                PiMarks[i] = PiMarks[faninId];
            }
            else {
                for (ll k = 0; k < net.GetPiNum(); k++) {
                    PiMarks[i][k] |= PiMarks[faninId][k]; 
                }
            }
        }
    }

    // generate divisor sets
    for (ll i = 0; i < vLInew.size(); ++i) {
        // level of pivot node
        ll pivLev = net.GetObjLev(vLInew[i]);
        // type 1: remove one node (node id is vLInew[i])
        vector <ll> div;
        for (ll j = 0; j < vLInew.size(); ++j) {
            if (j == i)
                continue;
            div.push_back(vLInew[j]);
        }
        if (div.size() == 0)
            continue;
        // if (div.size() > 1)     // at least 2
        //     vDiv.push_back(div);
        
        // type 2: substitute one node with its TFI
        bool fSuccess = false;  
        if (div.size() < 4) {
            net.SetNetNotTrav();
            Abc_NtkCleanMarkB(net.GetNet());
            Vec_Ptr_t * vCone = Abc_MfsWinMarkTfi(net.GetObj(vLI[i]));
            ll k;
            Abc_Obj_t * pDiv;
            Vec_PtrForEachEntry(Abc_Obj_t *, vCone, pDiv, k) {      // haven't limit the number
                pDiv->fMarkB = 1;
                if (std::find(vLI.begin(), vLI.end(), pDiv->Id) != vLI.end())   // exclude original inputs
                    continue;
                if (pDiv->fMarkA)
                    continue;
                if (pDiv->Level < pivLev - nlevLim)     // limit level
                    continue;
                if (std::find(div.begin(), div.end(), pDiv->Id) != div.end())   // avoid duplication 
                    continue;
                fSuccess = true;
                div.push_back(pDiv->Id);
                vDiv.push_back(div);
                div.pop_back();
            }

            // type 3: consider more nodes which are not in TFI cone
            ll nPi = net.GetPiNum();
            Abc_NtkForEachNode(net.GetNet(), pNode, k) {
                if (pNode->fMarkA || pNode->fMarkB)
                    continue;
                if (pNode->Level > pivLev)
                    continue;
                if (pNode->Level < pivLev - nlevLim)     // limit level
                    continue;
                // check support PI set
                bool fAdd = true;
                for (ll m = 0; m < nPi; m++) {
                    if (PiMarks[pNode->Id][m] && !PiMarks[vLI[i]][m]) {
                        fAdd = false;
                        break;
                    }
                }
                if (fAdd) {
                    fSuccess = true;
                    div.push_back(pNode->Id);
                    vDiv.push_back(div);
                    div.pop_back();
                }
            }
            Abc_NtkCleanMarkB(net.GetNet());
        }

        if (!fSuccess && (vLInew.size() > 4) && (div.size() > 1))
            vDiv.push_back(div);
    }
    
    if ((vLInew.size() <= 4) && (vLInew.size() >= 2)) {
        vDiv.push_back(vLInew);     // consider original LI set as a divisor set
    }

    Abc_NtkCleanMarkA(net.GetNet());

    bool fError = false;
    for (const auto & div : vDiv) {
        if (div.size() < 2) {
            fError = true;
            cout << "vLInew: ";
            for (const ll d : vLInew) {
                cout << d << ", ";
            }
            cout << "  div: ";
            for (const ll d : div) {
                cout << d << ", ";
            }
            cout << endl;
        }
    }
    assert(!fError);

    // remove redundant divisor sets
    simplifyDivs(vDiv);

    ll nMaxDivNum = 500;    // can be tuned
    if (vDiv.size() > nMaxDivNum)
        vDiv.resize(nMaxDivNum);

    return vDiv.size();
}

void SubCktMan::GenDivs() {
    ll levLim = 20;     // can be tuned
    cout << "parameter for GenDivs: levLim = " << levLim << endl;
    // evaluate #div
    ll minValue = std::numeric_limits<long long>::max();
    ll maxValue = 0;
    ll sum = 0;
    for (const auto& pSub : pSubCkts2) {
        ll num = pSub->GenDiv(net, levLim);
        minValue = min(minValue, num);
        maxValue = max(maxValue, num);
        sum += num;
    }
    cout << "for each SubCkt2, #div's min = " << minValue << ", max = " << maxValue << ", avg = " << static_cast<double>(sum)/static_cast<double>(pSubCkts2.size()) << endl;

    minValue = std::numeric_limits<long long>::max();
    maxValue = 0;
    sum = 0;
    for (const auto& pSub : pSubCkts3) {
        ll num = pSub->GenDiv(net, levLim);
        minValue = min(minValue, num);
        maxValue = max(maxValue, num);
        sum += num;
    }
    cout << "for each SubCkt3, #div's min = " << minValue << ", max = " << maxValue << ", avg = " << static_cast<double>(sum)/static_cast<double>(pSubCkts3.size()) << endl;
}

void SubCktMan::GenAllAppRWs(Simulator & appSmlt, Simulator & accSmlt, Database & db, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef, ll nThread) {
    // bool fBreak = false;
    // ll nLimit = 300;
    // ll vLoId = 0; 
    // for (const auto& pSub : pSubCkts2) {
    //     auto vLO = pSub->GetvLO();
    //     auto vDiv = pSub->GetvDiv();
    //     if (vDiv.size() == 0)
    //         continue;
    //     auto area = pSub->GetArea();

    //     // print sub-circuit
    //     cout << "i = " << vLoId << ": ";
    //     cout << "  LO: ";
    //     for (ll lo : vLO) {
    //         cout << lo << " ";
    //     }
    //     cout << "  LI: ";
    //     for (ll li : pSub->GetvLI()) {
    //         cout << li << " ";
    //     }
    //     cout << " area = " << pSub->GetArea() << ", #nodes = " << pSub->GetNodeNum() << ", #div = " << vDiv.size() << endl;

    //     for (const auto& div : vDiv) {
    //         if (GetAppRwNum() > nLimit) {
    //             fBreak = true;
    //             break;
    //         }
    //         GenAppRWPro(div, vLO, vLoId, appSmlt, accSmlt, db, area, bdPo2NodesRef);
    //     }
    //     ++vLoId;
    //     // cout << endl;
    // }

    // // pSubCkts3
    // vLoId = 0;
    // for (const auto& pSub : pSubCkts3) {
    //     auto vLO = pSub->GetvLO();
    //     auto vDiv = pSub->GetvDiv();
    //     auto area = pSub->GetArea();

    //     // print sub-circuit
    //     cout << "i = " << vLoId << ": ";
    //     cout << "  LO: ";
    //     for (ll lo : vLO) {
    //         cout << lo << " ";
    //     }
    //     cout << "  LI: ";
    //     for (ll li : pSub->GetvLI()) {
    //         cout << li << " ";
    //     }
    //     cout << " area = " << pSub->GetArea() << ", #nodes = " << pSub->GetNodeNum() << ", #div = " << vDiv.size() << endl;

    //     for (const auto& div : vDiv) {
    //         if (GetAppRwNum() > nLimit) {
    //             fBreak = true;
    //             break;
    //         }
    //         GenAppRWPro(div, vLO, vLoId, appSmlt, accSmlt, db, area, bdPo2NodesRef);
    //     }
    //     ++vLoId;
    //     // cout << endl;
    // }

    // if (fBreak)
    //     cout << "#AppRwNum > " << nLimit << ", break!" << endl;


    // multi-thread version
    // Merge all subcircuits
    std::vector<std::shared_ptr<SubCkt>> allSubCkts = pSubCkts2;
    allSubCkts.insert(allSubCkts.end(), pSubCkts3.begin(), pSubCkts3.end());
    ll total = allSubCkts.size();
    if (total == 0) return;

    ll realThread = std::min(nThread, total);
    cout << "realThread = " << realThread << endl;
    ll chunkSize = total / realThread;
    ll remainder = total % realThread;

    timer::progress_display pd(total);  // progress display
    std::vector<std::thread> threads;
    std::mutex progress_mutex;          // guards pd

    ll start = 0;
    for (ll t = 0; t < realThread; ++t) {
        ll end = start + chunkSize + (t < remainder ? 1 : 0);

        ll threadStart = start;
        ll threadEnd = end;

        threads.emplace_back([=, &appSmlt, &accSmlt, &db, &bdPo2NodesRef, &pd, &progress_mutex, this]() {
            for (ll i = threadStart; i < threadEnd; ++i) {
                auto& pSub = allSubCkts[i];
                auto vLO = pSub->GetvLO();
                auto vDiv = pSub->GetvDiv();
                if (vDiv.size() == 0)
                    continue;
                auto area = pSub->GetArea();

                ll vLoId = i;
                if (i >= pSubCkts2.size())
                    vLoId -= pSubCkts2.size();

                for (const auto& div : vDiv) {
                    this->GenAppRWPro(div, vLO, vLoId, appSmlt, accSmlt, db, area, bdPo2NodesRef, -1);
                }

                // thread-safe progress update
                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    ++pd;
                }
            }
        });

        start = end;
    }

    for (auto& t : threads) {
        t.join();
    }
}

struct TaskIndex {
    size_t subCkt_idx; // Index 'i' for allSubCkts
    size_t div_idx;    // Index 'j' for the vDiv vector
};

void SubCktMan::GenAllAppRWsPro(Simulator & appSmlt, Simulator & accSmlt, Database & db, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef, ll nThread) {
    //【preparation】
    pAppRWs2.resize(pSubCkts2.size());
    divInfo2.resize(pSubCkts2.size());
    hdRank2.resize(pSubCkts2.size());

    pAppRWs3.resize(pSubCkts3.size());
    divInfo3.resize(pSubCkts3.size());
    hdRank3.resize(pSubCkts3.size());

    //【prepare hamMarks & nIniFlipBit】
    // Lightweight Task Indexing (Instead of Task Flattening)
    std::vector<std::shared_ptr<SubCkt>> allSubCkts = pSubCkts2;
    allSubCkts.insert(allSubCkts.end(), pSubCkts3.begin(), pSubCkts3.end());
    if (allSubCkts.empty()) return;
    
    // Create a flat list of all tasks
    std::vector<TaskIndex> all_task_indices;
    size_t pSubCkts2_size = this->pSubCkts2.size();
    for (size_t i = 0; i < allSubCkts.size(); ++i) {
        const auto& pSub = allSubCkts[i];
        if (!pSub)
            assert(0);
        const auto& vDiv = pSub->GetvDiv();
        for (size_t j = 0; j < vDiv.size(); ++j) {
            all_task_indices.push_back({i, j});
        }
    }
    ll total_tasks = all_task_indices.size();
    if (total_tasks == 0) {
        std::cout << "No valid tasks to execute." << std::endl;
        return;
    }

    // Multithreaded Execution
    ll realThread = std::min(static_cast<ll>(nThread), total_tasks);
    std::cout << "PreGenAppRW: Total individual tasks: " << total_tasks << ", Using " << realThread << " threads." << std::endl;
    
    timer::progress_display pd(total_tasks);
    std::vector<std::thread> threads;
    std::mutex progress_mutex;
    std::atomic<ll> atomic_task_idx(0);

    auto startPreGenAppRW = chrono::system_clock::now();

    for (ll t = 0; t < realThread; ++t) {
        threads.emplace_back([&]() {
            ll idx;
            while ((idx = atomic_task_idx.fetch_add(1, std::memory_order_relaxed)) < total_tasks) {
                const TaskIndex& task_idx = all_task_indices[idx];
                size_t i = task_idx.subCkt_idx;
                size_t j = task_idx.div_idx;

                const auto& pSub = allSubCkts[i];
                auto vLO = pSub->GetvLO();
                auto div = pSub->GetvDiv()[j];
                
                ll vLoId = static_cast<ll>(i);
                if (i >= pSubCkts2_size) {
                    vLoId -= pSubCkts2_size;
                }

                this->PreGenAppRW(div, vLO, vLoId, appSmlt, j, bdPo2NodesRef);

                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    ++pd;
                }
            }
        });
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto duration = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - startPreGenAppRW);
    cout << "runtime for PreGenAppRWs = " << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " sec" << endl;

    //【search for appRWs】
    // sort by nIniFlipBits
    for (ll vLoId = 0; vLoId < pSubCkts2.size(); ++vLoId) {
        std::sort(divInfo2[vLoId].begin(), divInfo2[vLoId].end(), [](const DivInfo & a, const DivInfo & b) {return a.GetHd0() < b.GetHd0();});
        for (ll i = 0; i < divInfo2[vLoId].size(); ++i)
            divInfo2[vLoId][i].SetRank(i);
    }
    for (ll vLoId = 0; vLoId < pSubCkts3.size(); ++vLoId) {
        std::sort(divInfo3[vLoId].begin(), divInfo3[vLoId].end(), [](const DivInfo & a, const DivInfo & b) {return a.GetHd0() < b.GetHd0();});
        for (ll i = 0; i < divInfo3[vLoId].size(); ++i)
            divInfo3[vLoId][i].SetRank(i);
    }

    // Group tasks
    std::vector<std::vector<TaskIndex>> grouped_tasks(allSubCkts.size());
    for (size_t vLoId = 0; vLoId < pSubCkts2.size(); ++vLoId) {
        for (size_t divRank = 0; divRank < divInfo2[vLoId].size(); ++divRank) {     // due to the sorting by nIniFlipBits
            grouped_tasks[vLoId].push_back({vLoId, static_cast <size_t>(divInfo2[vLoId][divRank].GetDivId())});
        }
    }
    for (size_t vLoId = 0; vLoId < pSubCkts3.size(); ++vLoId) {
        for (size_t divRank = 0; divRank < divInfo3[vLoId].size(); ++divRank) {
            grouped_tasks[vLoId + pSubCkts2_size].push_back({vLoId + pSubCkts2_size, static_cast <size_t>(divInfo3[vLoId][divRank].GetDivId())});
        }
    }

    // Interleave tasks to create the final task queue
    all_task_indices.clear();
    bool tasks_left = true;
    while (tasks_left) {
        tasks_left = false;
        for (size_t i = 0; i < grouped_tasks.size(); ++i) {
            if (!grouped_tasks[i].empty()) {
                all_task_indices.push_back(grouped_tasks[i].front());
                grouped_tasks[i].erase(grouped_tasks[i].begin());
                tasks_left = true;
            }
        }
    }
    total_tasks = all_task_indices.size();

    // sort by divId again (for easy lookup of hamMarks & nIniFlipBits)
    for (ll vLoId = 0; vLoId < pSubCkts2.size(); ++vLoId)
        std::sort(divInfo2[vLoId].begin(), divInfo2[vLoId].end(), [](const DivInfo & a, const DivInfo & b) {return a.GetDivId() < b.GetDivId();});
    for (ll vLoId = 0; vLoId < pSubCkts3.size(); ++vLoId)
        std::sort(divInfo3[vLoId].begin(), divInfo3[vLoId].end(), [](const DivInfo & a, const DivInfo & b) {return a.GetDivId() < b.GetDivId();});

    // second Multithreaded Execution
    realThread = std::min(static_cast<ll>(nThread), total_tasks);
    std::cout << "GenAppRW: Total individual tasks: " << total_tasks << ", Using " << realThread << " threads." << std::endl;
    timer::progress_display pd2(total_tasks);
    std::vector<std::thread> threads2;
    std::mutex progress_mutex2;
    std::atomic<ll> atomic_task_idx2(0);

    auto startGenAppRW = chrono::system_clock::now();

    for (ll t = 0; t < realThread; ++t) {
        threads2.emplace_back([&]() {
            ll idx;
            while ((idx = atomic_task_idx2.fetch_add(1, std::memory_order_relaxed)) < total_tasks) {
                const TaskIndex& task_idx = all_task_indices[idx];
                size_t i = task_idx.subCkt_idx;
                size_t j = task_idx.div_idx;

                const auto& pSub = allSubCkts[i];
                auto vLO = pSub->GetvLO();
                auto div = pSub->GetvDiv()[j];
                const double area = pSub->GetArea();
                ll vLoId = static_cast<ll>(i);
                if (i >= pSubCkts2_size) {
                    vLoId -= pSubCkts2_size;
                }

                this->GenAppRWPro(div, vLO, vLoId, appSmlt, accSmlt, db, area, bdPo2NodesRef, j);

                {
                    std::lock_guard<std::mutex> lock(progress_mutex2);
                    ++pd2;
                }
            }
        });
    }

    for (auto& t : threads2) {
        if (t.joinable()) {
            t.join();
        }
    }

    duration = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - startGenAppRW);
    cout << "runtime for GenAppRWs = " << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " sec" << endl;

    // limit the num
    for (ll vLoId = 0; vLoId < pAppRWs2.size(); ++vLoId) {
        if (pAppRWs2[vLoId].size() > maxAppRWNum) {
            std::nth_element(pAppRWs2[vLoId].begin(), pAppRWs2[vLoId].begin() + maxAppRWNum, pAppRWs2[vLoId].end(), [](const std::shared_ptr<AppRW>& a, const std::shared_ptr<AppRW>& b) {return a->GetHd() > b->GetHd();});
            pAppRWs2[vLoId].resize(maxAppRWNum);
        }
    }
    for (ll vLoId = 0; vLoId < pAppRWs3.size(); ++vLoId) {
        if (pAppRWs3[vLoId].size() > maxAppRWNum) {
            std::nth_element(pAppRWs3[vLoId].begin(), pAppRWs3[vLoId].begin() + maxAppRWNum, pAppRWs3[vLoId].end(), [](const std::shared_ptr<AppRW>& a, const std::shared_ptr<AppRW>& b) {return a->GetHd() > b->GetHd();});
            pAppRWs3[vLoId].resize(maxAppRWNum);
        }
    }

    //【store in pAppRWs】
    for (const auto & pAppRWsForOneLoSet : pAppRWs2) {
        for (const auto & pAppRW : pAppRWsForOneLoSet) {
            pAppRWs.push_back(pAppRW);
        }
    }
    for (const auto & pAppRWsForOneLoSet : pAppRWs3) {
        for (const auto & pAppRW : pAppRWsForOneLoSet) {
            pAppRWs.push_back(pAppRW);
        }
    }
    
    std::cout << "\nGenAllAppRWsPro: Parallel processing finished." << std::endl;
}


void SubCktMan::GenAppRW(vector <ll> vDiv, vector <ll> vLO, ll vLoId, Simulator & appSmlt, Simulator & accSmlt, Database & db, double accArea, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef) {
    // cout << "#LO = " << vLO.size() << ", #Div = " << vDiv.size() << endl;
    // 1. build truth table between divisors and LOs (possibly infeasible)
    vector < vector < PattValue > > table((1LL << vDiv.size()), vector <PattValue>(vLO.size()));    // [divisor pattern][LO id]
    for (ll iFrame = 0; iFrame < appSmlt.GetFrameNumb(); ++iFrame) {
        // get divisor pattern
        ll divPatt = 0; 
        for (ll i = 0; i < vDiv.size(); ++i) {
        // for (ll i = vDiv.size() - 1; i >= 0 ; --i) {
            divPatt = divPatt * 2 + appSmlt.GetDat(vDiv[i], iFrame);
        }

        // check don't care pattern for LOs (need to consider the influence of multiple LOs on POs as a whole)
        ll nPo = net.GetPoNum();
        bool isDontCare = true;
        if (vLO.size() == 2) {
            if (vLO2Relation[vLoId] == 1) {
                for (ll o = 0; o < nPo; ++o) {
                    if (bdPo2Nodes11[o][vLoId][iFrame] || bdPo2Nodes10[o][vLoId][iFrame] || bdPo2NodesRef[o][vLO[1]][iFrame]) {
                        isDontCare = false;
                        break;
                    }                   
                }
            }
            else {
                for (ll o = 0; o < nPo; ++o) {
                    if (bdPo2Nodes11[o][vLoId][iFrame] || bdPo2NodesRef[o][vLO[0]][iFrame] || bdPo2NodesRef[o][vLO[1]][iFrame]) {
                        isDontCare = false;
                        break;
                    }                   
                }
            }
        }
        else if (vLO.size() == 3) {
            for (ll o = 0; o < nPo; ++o) {
                if (bdPo2Nodes101[o][vLoId][iFrame] || bdPo2Nodes110[o][vLoId][iFrame] || bdPo2Nodes011[o][vLoId][iFrame] || bdPo2Nodes111[o][vLoId][iFrame] || bdPo2NodesRef[o][vLO[0]][iFrame] || bdPo2NodesRef[o][vLO[1]][iFrame] || bdPo2NodesRef[o][vLO[2]][iFrame]) {
                    isDontCare = false;
                    break;
                }                   
            }
        }

        // assign value to n0/n1/d
        if (isDontCare) {
            for (ll o = 0; o < vLO.size(); ++o) {
                ++table[divPatt][o].d;
            }
        }
        else {
            for (ll o = 0; o < vLO.size(); ++o) {
                if (appSmlt.GetDat(vLO[o], iFrame))
                    ++table[divPatt][o].n1;
                else
                    ++table[divPatt][o].n0;
            }
        }
    }

    // cout << "finish build truth table between divisors and LOs" << endl;

    vector < FlatTtMark > tableMark(vLO.size() * (1LL << vDiv.size()));
    ll i = 0;
    for (ll o = 0; o < vLO.size(); ++o) {
        for (ll iPatt = 0; iPatt < (1LL << vDiv.size()); ++iPatt) {
            tableMark[i].LoId = o;
            tableMark[i].divPattId = iPatt;
            tableMark[i].mark = table[iPatt][o].n0 - table[iPatt][o].n1;
            ++i;
        }
    }
    sort(tableMark.begin(), tableMark.end(), [](const FlatTtMark & a, const FlatTtMark & b) {
        return abs(a.mark) < abs(b.mark);
    });

    // 2. choose feasible function
    ll exploreNumLim = 30;
    ll exploreNum = 0;

    ll nSuccess = 0;
    ll nAreaExceed = 0;
    ll nErrExceed = 0;

    // calculate initial feasible function with smallest hamming distance and checkRW
    ll nFlipBits = 0;
    vector <ll> vFeasibleTt(vLO.size(), 0);  // Tt: truth table
    for (ll o = 0; o < vLO.size(); ++o) {
        for (ll iPatt = 0; iPatt < (1LL << vDiv.size()); ++iPatt) {
            if (table[iPatt][o].n0 < table[iPatt][o].n1) {   // set the bit to 1
                vFeasibleTt[o] += (1LL << iPatt);
                nFlipBits += table[iPatt][o].n0;
            }
            else {
                nFlipBits += table[iPatt][o].n1;
            }
            // note: if n0 == n1, set the bit to 0 by default
        }
    }
    // cout << "initial feasible funcion: nFlipBits = " << nFlipBits << endl;

    cout << "initial feasible function = { ";
    for (const auto & func : vFeasibleTt)
        cout << func << " ";
    cout << "}" << endl;

    // if (vFeasibleTt[0] == 253 && vFeasibleTt[1] == 51) {
    //     cout << "#LO = " << vLO.size() << ", #Div = " << vDiv.size() << ", vFeasibleTt[0] == 253, vFeasibleTt[1] == 51" << endl;
    //     // print truth table
    //     cout << "print truth table: " << endl;
    //     for (ll iPatt = 0; iPatt < (1LL << vDiv.size()); ++iPatt) {
    //         cout << "divPatt = " << iPatt << ": ";
    //         for (ll o = 0; o < vLO.size(); ++o) {
    //             cout << "LO" << o;
    //             cout << "{" << table[iPatt][o].n0 << ", " << table[iPatt][o].n1 << ", " << table[iPatt][o].d << "}, ";
    //         }
    //         cout << endl;
    //     }
    //     cout << endl;
    //     // print tableMark
    //     cout << "print tableMark: " << endl;
    //     for (const auto & entry : tableMark) {
    //         cout << "divPattId = " << entry.divPattId << ", LoId = " << entry.LoId << ", n0 - n1 = " << entry.mark << endl;
    //     }
    // }

    // print truth table
    // cout << "print truth table: " << endl;
    // for (ll iPatt = 0; iPatt < (1LL << vDiv.size()); ++iPatt) {
    //     cout << "divPatt = " << iPatt << ": ";
    //     for (ll o = 0; o < vLO.size(); ++o) {
    //         cout << "LO" << o;
    //         cout << "{" << table[iPatt][o].n0 << ", " << table[iPatt][o].n1 << ", " << table[iPatt][o].d << "}, ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    int r = CheckRW_debug(vFeasibleTt, vDiv, vLO, vLoId, appSmlt, accSmlt, db, accArea, bdPo2NodesRef);
    cout << "return value = " << r << endl << endl;
    if (r == 0)
        ++nSuccess;
    else if (r == 1)
        ++nAreaExceed;
    else if (r == 2)
        ++nErrExceed;

    // derive next-best feasible functions
    ll flipBitNumLim = 15;
    priority_queue<Subset, vector<Subset>, greater<>> pq;
    for (ll i = 0; i < min(flipBitNumLim, static_cast <ll> (tableMark.size())); ++i) {
        pq.push({abs(tableMark[i].mark), {i}, i});
    }
    set<vector<ll>> visited;
    bool fContinue = true;
    while (fContinue) {
        // cout << "exploreNum = " << exploreNum << endl;
        // derive the next-best feasible function
        auto [sum, idxs, last] = pq.top();
        vector <ll> vFeasibleTtNew = vFeasibleTt;
        for (ll ii : idxs) {
            ll LoId = tableMark[ii].LoId;
            ll divPattId = tableMark[ii].divPattId;
            
            // cout << "flip: LoId = " << LoId << ", divPattId = " << divPattId << ", n0 - n1 = " << tableMark[ii].mark << endl;

            if (tableMark[ii].mark >= 0)    // n0 >= n1, need to turn the original 0 to 1
                vFeasibleTtNew[LoId] += (1LL << divPattId); 
            else    // n0 < n1, need to turn the original 1 to 0    
                vFeasibleTtNew[LoId] -= (1LL << divPattId);
        }

        // debug
        // for (ll o = 0; o < vLO.size(); ++o) {
        //     if (vFeasibleTtNew[o] < 0 || (vFeasibleTtNew[o] >= (1LL << (1LL << vDiv.size())))) {
        //         cout << "o = " << o << ", vFeasibleTtNew[o] = " << vFeasibleTtNew[o] << ", vFeasibleTt[o] = " << vFeasibleTt[o] << endl;
        //         for (ll ii : idxs) {
        //             cout << "LoId = " << tableMark[ii].LoId << ", divPattId = " << tableMark[ii].divPattId << ", tableMark[ii].mark = " << tableMark[ii].mark << endl;
        //         }
        //         assert(0);
        //     }
        // }

        // search feasibleTT in database and check the performance
        cout << "vFeasibleTtNew = { ";
        for (const auto & func : vFeasibleTtNew)
            cout << func << " ";
        cout << "}, ";
        r = CheckRW_debug(vFeasibleTtNew, vDiv, vLO, vLoId, appSmlt, accSmlt, db, accArea, bdPo2NodesRef);
        cout << "return value = " << r << endl << endl;
        if (r == 0)
            ++nSuccess;
        else if (r == 1)
            ++nAreaExceed;
        else if (r == 2)
            ++nErrExceed;
        ++exploreNum;

        // decide whether to continue
        if (exploreNum > exploreNumLim) {
            fContinue = false;
            if (nSuccess == 0 && nErrExceed < 10) {
                fContinue = true;
                exploreNumLim += 10;
            }
        }

        // Expand: try adding the next element (after the current one)
        pq.pop();
        for (ll j = last + 1; j < min(flipBitNumLim, static_cast <ll> (tableMark.size())); ++j) {
            vector<ll> newIdxs = idxs;
            newIdxs.push_back(j);
            if (visited.insert(newIdxs).second) {
                ll newSum = sum + abs(tableMark[j].mark);
                pq.push({newSum, newIdxs, j});
            }
        }

        if (pq.empty()) {
            break;
        }
        if (exploreNum >= 99)
            break;
    }
    cout << "nSuccess = " << nSuccess << ", nAreaExceed = " << nAreaExceed << ", nErrExceed = " << nErrExceed << endl;
    assert(0); 
}


void SubCktMan::PreGenAppRW(std::vector <ll> vDiv, std::vector <ll> vLO, ll vLoId, Simulator & appSmlt, ll divId, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef) {
    vector < vector < PattValue > > table((1LL << vDiv.size()), vector <PattValue>(vLO.size()));    // [divisor pattern][LO id]
    for (ll iFrame = 0; iFrame < nFrame; ++iFrame) {
        // get divisor pattern
        ll divPatt = 0; 
        for (ll i = 0; i < vDiv.size(); ++i) {
        // for (ll i = vDiv.size() - 1; i >= 0 ; --i) {
            divPatt = divPatt * 2 + appSmlt.GetDat(vDiv[i], iFrame);
        }

        // check don't care pattern for LOs (need to consider the influence of multiple LOs on POs as a whole)
        ll nPo = net.GetPoNum();
        bool isDontCare = true;
        if (vLO.size() == 2) {
            if (vLO2Relation[vLoId] == 1) {
                for (ll o = 0; o < nPo; ++o) {
                    if (bdPo2Nodes11[o][vLoId][iFrame] || bdPo2Nodes10[o][vLoId][iFrame] || bdPo2NodesRef[o][vLO[1]][iFrame]) {
                        isDontCare = false;
                        break;
                    }                   
                }
            }
            else {
                for (ll o = 0; o < nPo; ++o) {
                    if (bdPo2Nodes11[o][vLoId][iFrame] || bdPo2NodesRef[o][vLO[0]][iFrame] || bdPo2NodesRef[o][vLO[1]][iFrame]) {
                        isDontCare = false;
                        break;
                    }                   
                }
            }
        }
        else if (vLO.size() == 3) {
            for (ll o = 0; o < nPo; ++o) {
                if (bdPo2Nodes101[o][vLoId][iFrame] || bdPo2Nodes110[o][vLoId][iFrame] || bdPo2Nodes011[o][vLoId][iFrame] || bdPo2Nodes111[o][vLoId][iFrame] || bdPo2NodesRef[o][vLO[0]][iFrame] || bdPo2NodesRef[o][vLO[1]][iFrame] || bdPo2NodesRef[o][vLO[2]][iFrame]) {
                    isDontCare = false;
                    break;
                }                   
            }
        }
        else
            assert(0);

        // assign value to n0/n1/d
        if (isDontCare) {
            for (ll o = 0; o < vLO.size(); ++o) {
                ++table[divPatt][o].d;
            }
        }
        else {
            for (ll o = 0; o < vLO.size(); ++o) {
                if (appSmlt.GetDat(vLO[o], iFrame))
                    ++table[divPatt][o].n1;
                else
                    ++table[divPatt][o].n0;
            }
        }
    }

    vector <vector <ll>> hamMarks(vLO.size(), vector <ll> (1LL << vDiv.size()));  // [LoId][divPattId]
    for (ll o = 0; o < vLO.size(); ++o) {
        for (ll iPatt = 0; iPatt < (1LL << vDiv.size()); ++iPatt) {
            hamMarks[o][iPatt] = abs(table[iPatt][o].n0 - table[iPatt][o].n1);
        }
    }

    // calculate initial feasible function with smallest hamming distance and checkRW
    ll nIniFlipBits = 0;
    vector <ll> vIniFeasibleTt(vLO.size(), 0);  // Tt: truth table
    for (ll o = 0; o < vLO.size(); ++o) {
        for (ll iPatt = 0; iPatt < (1LL << vDiv.size()); ++iPatt) {
            if (table[iPatt][o].n0 < table[iPatt][o].n1) {   // set the bit to 1
                vIniFeasibleTt[o] += (1LL << iPatt);
                nIniFlipBits += table[iPatt][o].n0;
            }
            else {
                nIniFlipBits += table[iPatt][o].n1;
            }
            // note: if n0 == n1, set the bit to 0 by default
        }
    }

    if (vLO.size() == 2) {
        assert(vLoId < divInfo2.size());
        std::lock_guard<std::mutex> lock(mtx1);
        divInfo2[vLoId].emplace_back(divId, hamMarks, nIniFlipBits, vIniFeasibleTt);
    }
    else if (vLO.size() == 3) {
        assert(vLoId < divInfo3.size());
        std::lock_guard<std::mutex> lock(mtx2);
        divInfo3[vLoId].emplace_back(divId, hamMarks, nIniFlipBits, vIniFeasibleTt);
    }
    else {
        cout << "vLO.size() = " << vLO.size() << endl;
        assert(0);
    }
}


// main!
void SubCktMan::GenAppRWPro(vector <ll> vDiv, vector <ll> vLO, ll vLoId, Simulator & appSmlt, Simulator & accSmlt, Database & db, double accArea, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef, ll divId) {  
    ll nIniFlipBits = -1;
    if (vLO.size() == 2) {
        assert(divId < divInfo2[vLoId].size());
        assert(divId == divInfo2[vLoId][divId].GetDivId());
        nIniFlipBits = divInfo2[vLoId][divId].GetHd0();
    }
    else if (vLO.size() == 3) {
        assert(divId < divInfo3[vLoId].size());
        assert(divId == divInfo3[vLoId][divId].GetDivId());
        nIniFlipBits = divInfo3[vLoId][divId].GetHd0();
    }
    
    ll hdTh = GetHdTh(vLO.size(), vLoId);
    if (nIniFlipBits > hdTh) {
        return;
    }
    
    // db.SearchCands
    vector<Cand> cands;
    BdData bdData(bdPo2Nodes11, bdPo2Nodes10, bdPo2Nodes101,bdPo2Nodes110, bdPo2Nodes011, bdPo2Nodes111, bdPo2NodesRef, vLO2Relation);
    if (vLO.size() == 2)
        // db.SearchCands(vDiv.size(), vLO.size(), divInfo2[vLoId][divId].GetvIniFeasibleTt(), accArea, divInfo2[vLoId][divId].GetHamMarks(), cands, nIniFlipBits, appSmlt, bdData, hdTh - nIniFlipBits, maxAppRWNum, hdRank2[vLoId]);
        db.SearchCandsByAppFuncLib(vDiv.size(), vLO.size(), divInfo2[vLoId][divId].GetvIniFeasibleTt(), accArea, divInfo2[vLoId][divId].GetHamMarks(), cands, nIniFlipBits, appSmlt, hdTh - nIniFlipBits, maxAppRWNum, hdRank2[vLoId]);
    else if (vLO.size() == 3)
        // db.SearchCands(vDiv.size(), vLO.size(), divInfo3[vLoId][divId].GetvIniFeasibleTt(), accArea, divInfo3[vLoId][divId].GetHamMarks(), cands, nIniFlipBits, appSmlt, bdData, hdTh - nIniFlipBits, maxAppRWNum, hdRank3[vLoId]);
        db.SearchCandsByAppFuncLib(vDiv.size(), vLO.size(), divInfo3[vLoId][divId].GetvIniFeasibleTt(), accArea, divInfo3[vLoId][divId].GetHamMarks(), cands, nIniFlipBits, appSmlt, hdTh - nIniFlipBits, maxAppRWNum, hdRank3[vLoId]);

    for (const auto & cand : cands) {
        double reArea = accArea - cand.area;
        if (cand.area == -1)
            reArea = 0;
        auto pAppRW = std::make_shared<AppRW>(vLO, vDiv, cand.appFuncs, reArea, accArea, -1, (vLO.size() == 2) ? vLO2Relation[vLoId] : 0, vLoId, nIniFlipBits + cand.nFlipBits, (vLO.size() == 2) ? divInfo2[vLoId][divId].GetF0() : divInfo3[vLoId][divId].GetF0());
        if (vLO.size() == 2) {
            std::lock_guard<std::mutex> lock(mtx1);
            pAppRWs2[vLoId].push_back(pAppRW);
            hdRank2[vLoId].insert(nIniFlipBits + cand.nFlipBits);
        }
        else if (vLO.size() == 3) {
            std::lock_guard<std::mutex> lock(mtx2);
            pAppRWs3[vLoId].push_back(pAppRW);
            hdRank3[vLoId].insert(nIniFlipBits + cand.nFlipBits);
        }
        else {
            cout << "vLO.size() = " << vLO.size() << endl;
            assert(0);
        }
    }
}


int SubCktMan::CheckRW(std::vector <ll> vTt, std::vector <ll> vDiv, std::vector <ll> vLO, ll vLoId, Simulator & appSmlt, Simulator & accSmlt, Database & db, double accArea, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef) {
    // Remove duplicates
    set <ll> sTt(vTt.begin(), vTt.end());   // setTt
    
    // check whether the function exists in database
    set <ll> canoFuncs;
    vector <uint8_t> input_permutation;
    uint8_t input_inversion = 0;
    vector <uint8_t> output_inversion(sTt.size());
    bool fInputNpDiff = false;
    if (vDiv.size() == 2) {
        for (const auto & func: sTt) {
            if (!isFuncAreaOpt(func, 2))
                return 1;   // the rewriting is invalid because the area of the single-output function is too large
        }
        canoFuncs = sTt;
    }
    else {      // vDiv.size() == 3 or 4
        ll o = 0;
        for (const auto & func: sTt) {
            ll canoFunc = db.getNpnCanoFunc(func, vDiv.size());
            canoFuncs.insert(canoFunc);
            if (!isFuncAreaOpt(canoFunc, vDiv.size()))
                return 1;   // the rewriting is invalid because the area of the single-output function is too large
            if (o == 0) {
                input_permutation = db.getNpnPerm(func, vDiv.size());
                input_inversion = db.getNpnInv(func, vDiv.size()).first;
                output_inversion[o] = db.getNpnInv(func, vDiv.size()).second;
            }
            else {
                if (!fInputNpDiff) {
                    if (db.getNpnPerm(func, vDiv.size()) != input_permutation)
                        fInputNpDiff = true; 
                    else if (db.getNpnInv(func, vDiv.size()).first != input_inversion)
                        fInputNpDiff = true;   
                }
                output_inversion[o] = db.getNpnInv(func, vDiv.size()).second;                
            }
            ++o;
        }   // o (which is equal to sTt.size()) may not be equal to canoFuncs.size()
    }

    double winArea = 0;
    if (!fInputNpDiff) {
        vector <ll> vCanoFuncs(canoFuncs.begin(), canoFuncs.end());
        winArea = db.getWindowArea(vCanoFuncs, vDiv.size(), vCanoFuncs.size());
        if (winArea == -1) {
            cout << "winArea == -1! ";
            cout << "funcs: ";
            for (const auto & func: vTt) {
                cout << func << " ";
            }
            cout << "vCanoFuncs: ";
            for (const auto & func: vCanoFuncs) {
                cout << func << " ";
            }
            cout << endl;
            assert(winArea == -1);
        }
        
        if (vDiv.size() > 2) {
            ll nInv = 0;
            bitset<4> negIn(input_inversion);
            nInv += negIn.count();
            for (const auto & neg : output_inversion) {
                bitset<1> negOut(neg);
                nInv += negOut.count();
            }
            winArea += nInv * net.GetInvArea();
        }
    }
    else {  
        vector <ll> vTtPro(sTt.begin(), sTt.end());   // remove duplication    
        winArea = db.getWindowArea(vTtPro, vDiv.size(), vTtPro.size());
        if (winArea == -1) {
            // synthesis online
            winArea = SynthFunction_MultiOut(vTtPro, vDiv.size());

            // add data to online db
            db.InsertToMap(vTtPro, winArea, vDiv.size(), vTtPro.size());

            // mark for updating the offline db
            db.SetfUpdate(vDiv.size(), vTtPro.size());
        }
    }

    if (winArea >= accArea)
        return 1;   // the rewriting is invalid because the area of the rewriting circuit is not smaller than the area of the original sub-circuit

    // calculate area benifit
    double reArea = accArea - winArea;

    // calculate rewriting error
    double err = CalcRwErr(vTt, vDiv, vLO, vLoId, appSmlt, accSmlt, bdPo2NodesRef);

    if (err > errUppBound)
        return 2;

    vector <ll> vEmpty;
    auto pAppRW = std::make_shared<AppRW>(vLO, vDiv, vTt, reArea, accArea, err, (vLO.size() == 2) ? vLO2Relation[vLoId] : 0, vLoId, -1, vEmpty);
    pAppRWs.push_back(pAppRW);
    return 0;
}

int SubCktMan::CheckRW_debug(std::vector <ll> vTt, std::vector <ll> vDiv, std::vector <ll> vLO, ll vLoId, Simulator & appSmlt, Simulator & accSmlt, Database & db, double accArea, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef) {
    // calculate rewriting error
    double err = CalcRwErr(vTt, vDiv, vLO, vLoId, appSmlt, accSmlt, bdPo2NodesRef);
    cout << "err = " << err << endl;

    // Remove duplicates
    set <ll> sTt(vTt.begin(), vTt.end());   // setTt
    
    // check whether the function exists in database
    set <ll> canoFuncs;
    vector <uint8_t> input_permutation;
    uint8_t input_inversion = 0;
    vector <uint8_t> output_inversion(sTt.size());
    bool fInputNpDiff = false;
    if (vDiv.size() == 2) {
        for (const auto & func: sTt) {
            if (!isFuncAreaOpt(func, 2))
                return 1;   // the rewriting is invalid because the area of the single-output function is too large
        }
        canoFuncs = sTt;
    }
    else {      // vDiv.size() == 3 or 4
        ll o = 0;
        cout << "canoFuncs: {";
        for (const auto & func: sTt) {
            ll canoFunc = db.getNpnCanoFunc(func, vDiv.size());
            canoFuncs.insert(canoFunc);
            cout << canoFunc << " ";
            if (!isFuncAreaOpt(canoFunc, vDiv.size()))
                return 1;   // the rewriting is invalid because the area of the single-output function is too large
            if (o == 0) {
                input_permutation = db.getNpnPerm(func, vDiv.size());
                input_inversion = db.getNpnInv(func, vDiv.size()).first;
                output_inversion[o] = db.getNpnInv(func, vDiv.size()).second;
            }
            else {
                if (!fInputNpDiff) {
                    if (db.getNpnPerm(func, vDiv.size()) != input_permutation)
                        fInputNpDiff = true; 
                    else if (db.getNpnInv(func, vDiv.size()).first != input_inversion)
                        fInputNpDiff = true;   
                }
                output_inversion[o] = db.getNpnInv(func, vDiv.size()).second;                
            }
            ++o;
        }   // o (which is equal to sTt.size()) may not be equal to canoFuncs.size()
        cout << "} " << endl;
    }

    double winArea = 0;
    if (!fInputNpDiff) {
        vector <ll> vCanoFuncs(canoFuncs.begin(), canoFuncs.end());
        winArea = db.getWindowArea(vCanoFuncs, vDiv.size(), vCanoFuncs.size());
        if (winArea == -1) {
            cout << "winArea == -1! ";
            cout << "funcs: ";
            for (const auto & func: vTt) {
                cout << func << " ";
            }
            cout << "vCanoFuncs: ";
            for (const auto & func: vCanoFuncs) {
                cout << func << " ";
            }
            cout << endl;
            assert(winArea == -1);
        }
        
        if (vDiv.size() > 2) {
            ll nInv = 0;
            bitset<4> negIn(input_inversion);
            nInv += negIn.count();
            for (const auto & neg : output_inversion) {
                bitset<1> negOut(neg);
                nInv += negOut.count();
            }
            winArea += nInv * net.GetInvArea();
        }
    }
    else {  
        vector <ll> vTtPro(sTt.begin(), sTt.end());   // remove duplication    
        winArea = db.getWindowArea(vTtPro, vDiv.size(), vTtPro.size());
        if (winArea == -1) {
            // synthesis online
            winArea = SynthFunction_MultiOut(vTtPro, vDiv.size());

            // add data to online db
            db.InsertToMap(vTtPro, winArea, vDiv.size(), vTtPro.size());

            // mark for updating the offline db
            db.SetfUpdate(vDiv.size(), vTtPro.size());
        }
    }

    // calculate area benifit
    double reArea = accArea - winArea;

    // // calculate rewriting error
    // double err = CalcRwErr(vTt, vDiv, vLO, vLoId, appSmlt, accSmlt, bdPo2NodesRef);

    cout << "reArea = " << reArea << endl;

    if (winArea >= accArea)
        return 1;   // the rewriting is invalid because the area of the rewriting circuit is not smaller than the area of the original sub-circuit

    if (err > errUppBound)
        return 2;

    vector <ll> vEmpty;
    auto pAppRW = std::make_shared<AppRW>(vLO, vDiv, vTt, reArea, accArea, err, (vLO.size() == 2) ? vLO2Relation[vLoId] : 0, vLoId, -1, vEmpty);
    pAppRWs.push_back(pAppRW);
    return 0;
}

double SubCktMan::CalcRwErr(std::vector <ll> vTt, std::vector <ll> vDiv, std::vector <ll> vLO, ll vLoId, Simulator & appSmlt, Simulator & accSmlt, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef, ll nFrameHere) {
    if (nFrameHere == 0)
        nFrameHere = nFrame;
    ll nPo = appSmlt.GetPoNum();
    vector <boost::dynamic_bitset<ull>> newDat(nPo, boost::dynamic_bitset<ull>(nFrameHere));
    for (ll iPatt = 0; iPatt < nFrameHere; ++iPatt) {
        // obtain corresponding divisor pattern
        ll divPatt = 0; 
        for (ll i = 0; i < vDiv.size(); ++i) {
        // for (ll i = vDiv.size() - 1; i >= 0 ; --i) {
            divPatt = divPatt * 2 + appSmlt.GetDat(vDiv[i], iPatt);
        }
        // check whether each LO flips
        vector <int> flipMark(vLO.size(), 0);
        for (ll o = 0; o < vLO.size(); ++o) {   // traverse LO
            if (appSmlt.GetDat(vLO[o], iPatt) != ((vTt[o] >> divPatt) & 1))
                flipMark[o] = 1;
        }
        // cout << "1" << endl;
        // choose corresponding bd to calculate POs' new value
        const vector < vector < boost::dynamic_bitset <ull> > > * targetBd = nullptr;
        ll targetId = 0;
        bool fDoNotChange = false;
        if (vLO.size() == 2) {
            if (flipMark[0] && flipMark[1]) {
                targetBd = & bdPo2Nodes11;
                targetId = vLoId;
            }
            else if (flipMark[0] && (!flipMark[1])) {
                if (vLO2Relation[vLoId] == 1) {
                    targetBd = & bdPo2Nodes10;
                    targetId = vLoId;
                }
                else {
                    targetBd = & bdPo2NodesRef;
                    targetId = vLO[0];
                }
            }
            else if (!flipMark[0] && flipMark[1]) {
                targetBd = & bdPo2NodesRef;
                targetId = vLO[1];
            }
            else
                fDoNotChange = true;
        }
        else if (vLO.size() == 3) {
            if (flipMark[0] && flipMark[1] && flipMark[2]) {
                targetBd = & bdPo2Nodes111;
                targetId = vLoId;
            }
            else if (flipMark[0] && (!flipMark[1]) && flipMark[2]) {
                targetBd = & bdPo2Nodes101;
                targetId = vLoId;
            }
            else if (flipMark[0] && flipMark[1] && (!flipMark[2])) {
                targetBd = & bdPo2Nodes110;
                targetId = vLoId;
            }
            else if ((!flipMark[0]) && flipMark[1] && flipMark[2]) {
                targetBd = & bdPo2Nodes011;
                targetId = vLoId;
            }
            else if (flipMark[0] && (!flipMark[1]) && (!flipMark[2])) {
                targetBd = & bdPo2NodesRef;
                targetId = vLO[0];
            }
            else if ((!flipMark[0]) && flipMark[1] && (!flipMark[2])) {
                targetBd = & bdPo2NodesRef;
                targetId = vLO[1];
            }
            else if ((!flipMark[0]) && (!flipMark[1]) && flipMark[2]) {
                targetBd = & bdPo2NodesRef;
                targetId = vLO[2];
            }
            else
                fDoNotChange = true;
        }
        // cout << "2" << endl;
        // calculate POs' new value
        if (fDoNotChange) {
            for (ll o = 0; o < nPo; ++o) {
                auto poId = appSmlt.GetPoId(o);
                newDat[o][iPatt] = appSmlt.GetDat(poId, iPatt);
            }
        }
        else {
            for (ll o = 0; o < nPo; ++o) {
                auto poId = appSmlt.GetPoId(o);
                newDat[o][iPatt] = appSmlt.GetDat(poId, iPatt) ^ (*targetBd)[o][targetId][iPatt];
            }
        }
        // cout << "3" << endl;
    }

    // debug
    bool fDebug = false;
    // if (vLO.size() == 2) {
    //     if (vLO[0] == 51 && vLO[1] == 76 && vTt[0] == 7 && vTt[1] == 23) {
    //         fDebug = true;
    //     }
    // }

    // calculate error between POs' new value and acc value
    std::vector < boost::dynamic_bitset <ull> > accDat;
    accDat.resize(nPo);
    for (ll o = 0; o < nPo; ++o) {
        auto poId = accSmlt.GetPoId(o);
        accDat[o] = *(accSmlt.GetDat(poId));    // size if full nFrame!
    }
    return GetErrFromPoValue(accDat, newDat, isSign, nOutput, metrType, fDebug, errUppBound);
}


void SubCktMan::PrintAppRWs() {
    cout << "PrintAppRWs: " << endl;
    for (const auto & pAppRW: pAppRWs) {
        cout << "error = " << pAppRW->GetError() << ", reArea = " << pAppRW->GetReArea() << endl;
    }
}


std::shared_ptr <AppRW> SubCktMan::SelectBestAppRW(double backErr) {
    // std::sort(pAppRWs.begin(), pAppRWs.end(), [backErr](const std::shared_ptr<AppRW>& a, const std::shared_ptr<AppRW>& b) {
    //     constexpr double epsilon = 1e-6;  // Prevent division by zero
        
    //     double scoreA = a->GetReArea() / (std::abs((a->GetError()) - backErr) + epsilon);
    //     double scoreB = b->GetReArea() / (std::abs((b->GetError()) - backErr) + epsilon);

    //     // Special case: if error is smaller than backErr (i.e., more desirable), give a bonus
    //     if (a->GetError() - backErr < 0) scoreA += 1e6;
    //     if (b->GetError() - backErr < 0) scoreB += 1e6;

    //     return scoreA > scoreB;  // Higher score is better
    // });

    sort(pAppRWs.begin(), pAppRWs.end(), [](const std::shared_ptr<AppRW>& a, const std::shared_ptr<AppRW>& b) {
        // if (a->GetReArea() != b->GetReArea())
        //     return a->GetReArea() > b->GetReArea();
        // return a->GetError() < b->GetError();
        if (a->GetError() != b->GetError())
            return a->GetError() < b->GetError();
        return a->GetReArea() > b->GetReArea();
    });

    return pAppRWs[0];
}

void SubCktMan::SortAppRWs() {
    sort(pAppRWs.begin(), pAppRWs.end(), [](const std::shared_ptr<AppRW>& a, const std::shared_ptr<AppRW>& b) {
        if (a->GetError() != b->GetError())
            return a->GetError() < b->GetError();
        return a->GetReArea() > b->GetReArea();
    });

    // pAppRWs.erase(
    //     std::remove_if(pAppRWs.begin(), pAppRWs.end(),
    //         [this](const std::shared_ptr<AppRW>& p) {
    //             return p->GetReArea() <= 0 || p->GetError() > this->errUppBound;
    //         }),
    //     pAppRWs.end());
    // cout << "sort " << pAppRWs.size() << " appRWs" << endl;

    // // calculate score
    // for (ll i = 0; i < pAppRWs.size(); ++i) {
    //     auto pAppRW = pAppRWs[i];
    //     double score = (pAppRW->GetError() - backErr)/pAppRW->GetReArea();
    //     pAppRW->SetScore(score);
    // }

    // // sort by score (smaller is better)
    // sort(pAppRWs.begin(), pAppRWs.end(), [](const std::shared_ptr<AppRW>& a, const std::shared_ptr<AppRW>& b) {
    //     return a->GetScore() < b->GetScore();
    // });
}

void SubCktMan::SortAppRWsByErr() {
    sort(pAppRWs.begin(), pAppRWs.end(), [](const std::shared_ptr<AppRW>& a, const std::shared_ptr<AppRW>& b) {
        if (a->GetError() != b->GetError())
            return a->GetError() < b->GetError();
        return a->GetReArea() > b->GetReArea();
    });
    // remove RW whose error exceeds bound
    ll cutId = 0;
    for (const auto & pAppRW : pAppRWs) {
        if (pAppRW->GetError() > errUppBound)
            break;
        ++cutId;
    }
    pAppRWs.resize(cutId);
}


void AppRW::Print() {
    cout << "error = " << error << ", oriArea = " << oriArea << ", reArea = " << reArea << endl;
    cout << "LOs: ";
    for (const auto & LoId : vLO)
        cout << LoId << " ";
    cout << endl;

    cout << "divisors: ";
    for (const auto & divId : vDiv)
        cout << divId << " ";
    cout << endl;

    cout << "initial feasible function: ";
    for (const auto & func : f0)
        cout << func << " ";
    cout << endl;

    cout << "appFuncs: ";
    for (const auto & func : appFunc)
        cout << func << " ";
    cout << endl;

    if (vLO.size() == 2)
        cout << "fRelation of 2 LOs = " << fRelation << endl;
}

void SubCktMan::CleanForTrivialCase() {
    pSubCkts2 = pSubCkts2Trivial;
    pSubCkts3 = pSubCkts3Trivial;

    vLO2Relation.clear();
    bdPo2Nodes11.clear();
    bdPo2Nodes10.clear();
    bdPo2Nodes101.clear();
    bdPo2Nodes110.clear();
    bdPo2Nodes011.clear();
    bdPo2Nodes111.clear();
}


std::vector<ll> getUniqueSorted(std::vector<ll> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

std::vector<ll> mergeSortedDivs(const std::vector<ll>& div1, const std::vector<ll>& div2) {
    std::vector<ll> merged;
    // Reserve space to avoid multiple reallocations, estimating max possible size
    merged.reserve(div1.size() + div2.size());
    std::set_union(div1.begin(), div1.end(),
                   div2.begin(), div2.end(),
                   std::back_inserter(merged));
    return merged;
}

void simplifyDivs(std::vector<std::vector<ll>>& vDiv) {
    // 1. Pre-process each inner 'div': sort elements and remove duplicates.
    //    This ensures consistency and prepares them for efficient set operations.
    for (auto& div : vDiv) {
        div = getUniqueSorted(div);
    }

    // 2. Sort the entire 'vDiv'. This is an optional but highly recommended step.
    //    Sorting helps group similar 'divs' together, which can improve the
    //    effectiveness of the greedy merging strategy.
    //    The sorting criteria here are: first by the first element, then by size,
    //    then by lexicographical comparison (full content) as a tie-breaker.
    std::sort(vDiv.begin(), vDiv.end(), [](const std::vector<ll>& a, const std::vector<ll>& b) {
        // Handle empty vectors first for robust comparison
        if (a.empty() && b.empty()) return false;
        if (a.empty()) return true;  // Empty 'a' comes before non-empty 'b'
        if (b.empty()) return false; // Non-empty 'a' comes after empty 'b'

        // Primary sort: by the first element
        if (a[0] != b[0]) return a[0] < b[0];
        // Secondary sort: by size (smaller 'divs' first)
        if (a.size() != b.size()) return a.size() < b.size();
        // Tertiary sort: lexicographical comparison (full content)
        return a < b;
    });

    // Create a temporary vector to store the simplified 'divs'.
    // We can't modify 'vDiv' directly while iterating and removing elements efficiently.
    std::vector<std::vector<ll>> tempResultDivs;
    if (vDiv.empty()) {
        return; // Nothing to simplify if input is empty
    }

    // Add the first pre-processed 'div' to the result.
    tempResultDivs.push_back(vDiv[0]);

    // Iterate through the remaining 'divs' to attempt merging.
    for (size_t i = 1; i < vDiv.size(); ++i) {
        const auto& currentDiv = vDiv[i];
        bool merged = false;

        // Try to merge 'currentDiv' with the last 'div' in our temporary result.
        // This is a greedy approach.
        if (!tempResultDivs.empty()) {
            std::vector<ll> potentialMergedDiv = mergeSortedDivs(tempResultDivs.back(), currentDiv);

            // Check if the merged 'div' respects the size constraint (<= 4).
            if (potentialMergedDiv.size() <= 4) {
                // If it fits, update the last 'div' in the result with the merged version.
                tempResultDivs.back() = potentialMergedDiv;
                merged = true;
            }
        }

        // If 'currentDiv' could not be merged with the last one (either no merge was
        // attempted because result was empty, or the merged size exceeded 4),
        // add 'currentDiv' as a new, separate 'div' to the result.
        if (!merged) {
            tempResultDivs.push_back(currentDiv);
        }
    }

    // After processing all 'divs', replace the original 'vDiv' content
    // with the simplified result.
    vDiv = std::move(tempResultDivs);
}

ll SubCktMan::GetSubCktRank1(ll nLo, ll vLoId) const {
    if (nLo == 2) {
        assert(vLoId >= 0);
        assert(vLoId < pSubCkts2.size());
        return pSubCkts2[vLoId]->GetRank1();
    }
    else if (nLo == 3) {
        assert(vLoId >= 0);
        assert(vLoId < pSubCkts3.size());
        return pSubCkts3[vLoId]->GetRank1();
    }
    else
        assert(0);
}

ll SubCktMan::GetSubCktRank2(ll nLo, ll vLoId) const {
    if (nLo == 2) {
        assert(vLoId >= 0);
        assert(vLoId < pSubCkts2.size());
        return pSubCkts2[vLoId]->GetRank2();
    }
    else if (nLo == 3) {
        assert(vLoId >= 0);
        assert(vLoId < pSubCkts3.size());
        return pSubCkts3[vLoId]->GetRank2();
    }
    else
        assert(0);
}

void SubCktMan::BatchErr(Simulator & appSmlt, Simulator & accSmlt, const std::vector<std::vector<boost::dynamic_bitset<ull>>>& bdPo2NodesRef, ll nThread) {
    // 1. Preparation
    if (this->pAppRWs.empty()) {
        std::cout << "pAppRWs is empty, no tasks to execute." << std::endl;
        return;
    }

    // 2. Multithreaded Execution based on Index
    auto startBatch1 = chrono::system_clock::now();
    ll total_tasks = this->pAppRWs.size();
    ll realThread = std::min(static_cast<ll>(nThread), total_tasks);
    
    std::cout << "BatchErr: Total individual tasks: " << total_tasks << ", Using " << realThread << " threads." << std::endl;

    timer::progress_display pd(total_tasks);
    std::vector<std::thread> threads;
    std::mutex progress_mutex;
    std::atomic<ll> atomic_task_idx(0);

    ll smallFrame = 10048;
    cout << "use smallFrame = " << smallFrame << endl;

    for (ll t = 0; t < realThread; ++t) {
        threads.emplace_back([&]() {
            ll idx;
            // The loop continues as long as there are tasks left to process.
            while ((idx = atomic_task_idx.fetch_add(1, std::memory_order_relaxed)) < total_tasks) {
                // Here, 'idx' is the single index for the pAppRWs vector.
                
                // Execute the task for the current index.
                // Replace this with the actual operation you need to perform on pAppRWs[idx].
                // For example:
                // pAppRWs[idx]->SomeMethod();
                // Or:
                // this->ProcessAppRW(pAppRWs[idx]);

                // Example: print index
                // std::cout << "Processing task at index: " << idx << std::endl;

                auto pAppRW = pAppRWs[idx];
                double err = CalcRwErr(pAppRW->GetFuncs(), pAppRW->GetvDiv(), pAppRW->GetvLO(), pAppRW->GetvLoId(), appSmlt, accSmlt, bdPo2NodesRef, smallFrame);
                pAppRW->SetError(err);
                
                // Safely update the progress bar
                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    ++pd;
                }
            }
        });
    }

    // Wait for all threads to complete.
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto duration = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - startBatch1);
    cout << "runtime for BatchAppRWErr(small nFrame) = " << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " sec" << endl;

    // filter
    ll maxNum = 20000;   // can be tuned
    ll newSize = min(maxNum, static_cast<ll>(pAppRWs.size()));
    std::nth_element(pAppRWs.begin(), pAppRWs.begin() + newSize, pAppRWs.end(), [](const std::shared_ptr<AppRW>& a, const std::shared_ptr<AppRW>& b) {
        if (a->GetError() != b->GetError())
            return a->GetError() < b->GetError();
        return a->GetReArea() > b->GetReArea();
    });
    pAppRWs.resize(newSize);
    cout << "after filtering, #appRW = " << pAppRWs.size() << endl;

    // second muti-thread
    auto startBatch2 = chrono::system_clock::now();
    total_tasks = this->pAppRWs.size();
    if (total_tasks == 0) {
        std::cout << "After filtering, pAppRWs is empty. No second pass needed." << std::endl;
        return;
    }
    realThread = std::min(static_cast<ll>(nThread), total_tasks);
    
    std::cout << "BatchErr: Starting second pass. Total individual tasks: " << total_tasks << ", Using " << realThread << " threads." << std::endl;

    timer::progress_display pd2(total_tasks);
    std::vector<std::thread> threads2;
    std::mutex progress_mutex2;
    std::atomic<ll> atomic_task_idx2(0);

    for (ll t = 0; t < realThread; ++t) {
        threads2.emplace_back([&]() {
            ll idx;
            while ((idx = atomic_task_idx2.fetch_add(1, std::memory_order_relaxed)) < total_tasks) {
                auto pAppRW = pAppRWs[idx];
                
                // TODO: Put the second pass task here.
                // Replace the following line with the actual function call for the second pass.
                // For example:
                // pAppRW->RefineData(); 
                double err = CalcRwErr(pAppRW->GetFuncs(), pAppRW->GetvDiv(), pAppRW->GetvLO(), pAppRW->GetvLoId(), appSmlt, accSmlt, bdPo2NodesRef, 0);    // use full nFrame
                pAppRW->SetError(err);
                
                {
                    std::lock_guard<std::mutex> lock(progress_mutex2);
                    ++pd2;
                }
            }
        });
    }

    for (auto& t : threads2) {
        if (t.joinable()) {
            t.join();
        }
    }
    duration = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - startBatch2);
    cout << "runtime for BatchAppRWErr(big nFrame) = " << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " sec" << endl;
}

ll SubCktMan::GetHdTh(ll nLo, ll vLoId) {  
    // std::lock_guard<std::mutex> lock(appRwMutex);
    // if (nLo == 2) {
    //     assert(vLoId < pSubCkts2.size());
    //     if (pAppRWs2[vLoId].size() < maxAppRWNum) {
    //         return std::numeric_limits<long long>::max();
    //     }
    //     else {
    //         std::nth_element(pAppRWs2[vLoId].begin(), pAppRWs2[vLoId].begin() + maxAppRWNum, pAppRWs2[vLoId].end(), [](const std::shared_ptr<AppRW>& a, const std::shared_ptr<AppRW>& b) {return a->GetHd() > b->GetHd();});
    //         std::sort(pAppRWs2[vLoId].begin(), pAppRWs2[vLoId].begin() + maxAppRWNum, [](const std::shared_ptr<AppRW>& a, const std::shared_ptr<AppRW>& b) {return a->GetHd() > b->GetHd();});
    //         return pAppRWs2[vLoId][maxAppRWNum - 1]->GetHd();
    //     }
    // }
    // else if (nLo == 3) {
    //     assert(vLoId < pSubCkts3.size());
    //     if (pAppRWs3[vLoId].size() < maxAppRWNum) {
    //         return std::numeric_limits<long long>::max();
    //     }
    //     else {
    //         std::nth_element(pAppRWs3[vLoId].begin(), pAppRWs3[vLoId].begin() + maxAppRWNum, pAppRWs3[vLoId].end(), [](const std::shared_ptr<AppRW>& a, const std::shared_ptr<AppRW>& b) {return a->GetHd() > b->GetHd();});
    //         std::sort(pAppRWs3[vLoId].begin(), pAppRWs3[vLoId].begin() + maxAppRWNum, [](const std::shared_ptr<AppRW>& a, const std::shared_ptr<AppRW>& b) {return a->GetHd() > b->GetHd();});
    //         return pAppRWs3[vLoId][maxAppRWNum - 1]->GetHd();
    //     }
    // }
    // else {
    //     cout << "nLo = " << nLo << endl;
    //     assert(0);
    // }

    if (nLo == 2) {
        if (hdRank2[vLoId].size() < maxAppRWNum) {
            return std::numeric_limits<long long>::max();
        }
        else {
            auto it = hdRank2[vLoId].begin();
            std::advance(it, maxAppRWNum - 1);
            return *it;
        }
    }
    else if (nLo == 3) {
        if (hdRank3[vLoId].size() < maxAppRWNum) {
            return std::numeric_limits<long long>::max();
        }
        else {
            auto it = hdRank3[vLoId].begin();
            std::advance(it, maxAppRWNum - 1);
            return *it;
        }
    }
}