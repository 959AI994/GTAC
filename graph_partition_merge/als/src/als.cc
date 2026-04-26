#include "als.h"
#include "base/abc/abc.h"
#include <ostream>

using namespace std;
using namespace abc;
using namespace boost;

void ALSOpt::Print() {
    cout << "***** options" << endl;
    cout << (isSign? "use signed output": "use unsigned output") << endl;
    cout << "source seed = " << sourceSeed << endl;
    cout << "distrType = " << distrType << endl;
    cout << "metrType = " << metrType << endl;
    cout << "#simulation patterns = " << nFrame << endl;
    cout << "number of threads = " << nThread << endl;
    cout << metrType << " upper bound = " << errUppBound << endl;
    cout << "output path: " << outpPath << endl;
}


ALSMan::ALSMan(ALSOpt & opt): isSign(opt.isSign), sourceSeed(opt.sourceSeed), seed(opt.sourceSeed), distrType(opt.distrType), metrType(opt.metrType), nFrame(opt.nFrame), nOutput(opt.nOutput),  nThread(opt.nThread), errUppBound(opt.errUppBound), accNet(NetMan(opt.pNtk, true)), appCktName(opt.appCktName), outpPath(opt.outpPath), accCktName(opt.accCktName) {
    // randGen.seed(sourceSeed);
    // seed = NewSeed();
    randGen.seed(static_cast<unsigned int>(time(0)));
    if (accNet.GetNetType() == NET_TYPE::GATE) {
        accNet.ReArrInTopoOrd();
        maxDelay = accNet.GetDelay();
        maxDelayOri = maxDelay;
    }
    else   
        maxDelay = numeric_limits <double>::max();

    if (opt.pAppNtk == nullptr)
        pAppNtk = nullptr;
    else {
        pAppNtk = Abc_NtkDup(opt.pAppNtk);
    }
}


double ALSMan::ReplaceSubcircuit(ll index, const std::vector<std::pair<ll, std::string>> & subcktFiles, const std::unordered_map<ll, SubcktInfo> & subcktInfoMap, NetMan & Net, bool fPrint) {
    // Bounds-check index
    if (index < 0 || index >= static_cast<ll>(subcktFiles.size())) {
        cout << "Error: invalid subcircuit index " << index << endl;
        assert(0);
    }
    
    ll subId = subcktFiles[index].first;
    const std::string & filePath = subcktFiles[index].second;
    
    // get LO and LI from partition_map.txt
    auto it = subcktInfoMap.find(subId);
    if (it == subcktInfoMap.end()) {
        cout << "Error: subcircuit ID " << subId << " not found in partition_map.txt" << endl;
        assert(0);
    }
    const SubcktInfo & info = it->second;
    const std::vector<ll> & LO_ids = info.LO_ids;
    const std::vector<ll> & LI_ids = info.LI_ids;
    
    if (fPrint) {
        cout << "Reading subcircuit " << subId << ": " << filePath << endl;
        cout << "  LO (" << LO_ids.size() << "): ";
        for (ll id : LO_ids) cout << id << " ";
        cout << endl;
        cout << "  LI (" << LI_ids.size() << "): ";
        for (ll id : LI_ids) cout << id << " ";
        cout << endl;
    }
    
    std::string cmd = "read " + filePath + "; logic; sop;";
    Abc_Frame_t * pAbc = Abc_FrameGetGlobalFrame();
    assert(!Cmd_CommandExecute(pAbc, cmd.c_str()));
    Abc_Ntk_t * pSubNtk = Abc_NtkDup(Abc_FrameReadNtk(pAbc));

    // ref:
    // Abc_FrameReplaceCurrentNetwork(pAbc, Abc_NtkDup(pSubNtk));
    // string Command = string("strash;");
    // assert(!Cmd_CommandExecute(pAbc, Command.c_str()));
    // Abc_Ntk_t * pSynNtk = Abc_NtkDup(Abc_FrameReadNtk(pAbc));
    
    // [do substitution using LO_ids and LI_ids]
    vector <ll> vLO, vLI;
    for (ll id : LO_ids) {
        vLO.push_back(Net.GetNewId(id));
    }
    for (ll id : LI_ids) {
        vLI.push_back(Net.GetNewId(id));
    }
    Net.ReplaceSubCktPro(vLI, vLO, pSubNtk, LO_ids);
    
    auto netTmp = Net;  // avoid inconsistent node Id
    netTmp.IsOriIdAll0("netTmp");
    Net = netTmp;
    Net.IsOriIdAll0("after duplicating");
    Net.UpdateNewIdMap();
    // cout << "oriId 12176 corresponds to newId: " << currNet.GetNewId(12176) << endl;
    // currNet.ReArrInTopoOrd();
    // cout << "finish topo" << endl;
    // currNet.UpdateNewIdMap();
    // cout << "oriId 12176 corresponds to newId: " << currNet.GetNewId(12176) << endl;

    // evaluation
    vector <ll> RealCom;
    for (auto i = 0; i < nOutput; ++i)
        RealCom.emplace_back(0);    // set compensation to 0
    assert(Net.GetPoNum() % nOutput == 0);
    double err = CalcErr(accNet, Net, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
    // double err = -1;
    cout << "after substituting subcircuit " << subId << ", " << metrType << " = " << err << ", size = " << Net.GetArea() << ", depth = " << Net.GetDelay() << endl; 
    // cout << "after substituting subcircuit " << subId << ", " << metrType << " = " << err << ", size = " << currNet.GetArea() << endl;     
    
    return err;
}


void ALSMan::ReplaceSubcircuit_v2(ll index, const std::vector<std::pair<ll, std::string>> & subcktFiles, const std::unordered_map<ll, SubcktInfo> & subcktInfoMap, NetMan & Net, bool fPrint) {
    // Bounds-check index
    if (index < 0 || index >= static_cast<ll>(subcktFiles.size())) {
        cout << "Error: invalid subcircuit index " << index << endl;
        assert(0);
    }
    
    ll subId = subcktFiles[index].first;
    const std::string & filePath = subcktFiles[index].second;
    
    // get LO and LI from partition_map.txt
    auto it = subcktInfoMap.find(subId);
    if (it == subcktInfoMap.end()) {
        cout << "Error: subcircuit ID " << subId << " not found in partition_map.txt" << endl;
        assert(0);
    }
    const SubcktInfo & info = it->second;
    const std::vector<ll> & LO_ids = info.LO_ids;
    const std::vector<ll> & LI_ids = info.LI_ids;
    
    if (fPrint) {
        cout << "Reading subcircuit " << subId << ": " << filePath << endl;
        cout << "  LO (" << LO_ids.size() << "): ";
        for (ll id : LO_ids) cout << id << " ";
        cout << endl;
        cout << "  LI (" << LI_ids.size() << "): ";
        for (ll id : LI_ids) cout << id << " ";
        cout << endl;
    }
    
    std::string cmd = "read " + filePath + "; logic; sop;";
    Abc_Frame_t * pAbc = Abc_FrameGetGlobalFrame();
    assert(!Cmd_CommandExecute(pAbc, cmd.c_str()));
    Abc_Ntk_t * pSubNtk = Abc_NtkDup(Abc_FrameReadNtk(pAbc));

    // ref:
    // Abc_FrameReplaceCurrentNetwork(pAbc, Abc_NtkDup(pSubNtk));
    // string Command = string("strash;");
    // assert(!Cmd_CommandExecute(pAbc, Command.c_str()));
    // Abc_Ntk_t * pSynNtk = Abc_NtkDup(Abc_FrameReadNtk(pAbc));
    
    // [do substitution using LO_ids and LI_ids]
    vector <ll> vLO, vLI;
    for (ll id : LO_ids) {
        vLO.push_back(Net.GetNewId(id));
    }
    for (ll id : LI_ids) {
        vLI.push_back(Net.GetNewId(id));
    }
    Net.ReplaceSubCktPro(vLI, vLO, pSubNtk, LO_ids);
    
    auto netTmp = Net;  // avoid inconsistent node Id
    netTmp.IsOriIdAll0("netTmp");
    Net = netTmp;
    Net.IsOriIdAll0("after duplicating");
    Net.UpdateNewIdMap();
    // cout << "oriId 12176 corresponds to newId: " << currNet.GetNewId(12176) << endl;
    // currNet.ReArrInTopoOrd();
    // cout << "finish topo" << endl;
    // currNet.UpdateNewIdMap();
    // cout << "oriId 12176 corresponds to newId: " << currNet.GetNewId(12176) << endl;

    // evaluation
    // vector <ll> RealCom;
    // for (auto i = 0; i < nOutput; ++i)
    //     RealCom.emplace_back(0);    // set compensation to 0
    // assert(Net.GetPoNum() % nOutput == 0);
    // double err = CalcErr(accNet, Net, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
    // // double err = -1;
    // cout << "after substituting subcircuit " << subId << ", " << metrType << " = " << err << ", size = " << Net.GetArea() << ", depth = " << Net.GetDelay() << endl; 
    // // cout << "after substituting subcircuit " << subId << ", " << metrType << " = " << err << ", size = " << currNet.GetArea() << endl;     
    
    // return err;
}


void ALSMan::GraphMerge() {
    cout << "Graph merge begin!" << endl;
    if (accNet.GetNetType() == NET_TYPE::SOP) {
        cout << "SOP network" << endl;
    }
    else if (accNet.GetNetType() == NET_TYPE::AIG) {
        cout << "AIG network" << endl;
    }
    else if (accNet.GetNetType() == NET_TYPE::STRASH) {
        cout << "STRASH network" << endl;
    }
    accNet.ConvToSop();
    auto currNet = accNet;
    ll realPoNum = Abc_NtkPoNum(currNet.GetNet());
    // accNet.PrintPro(1, 1, 0); 
    vector <double> vErr;

    // [obtain subcktPath]
    // subcktPath is the ACT_out folder in the parent directory of outpPath
    // e.g., if outpPath is graph_merge/EPFL/random_control/arbiter/merge_out
    // then subcktPath is graph_merge/EPFL/random_control/arbiter/ACT_out
    // Note: outpPath may have trailing slash due to FixPath()
    std::string outpPathTrimmed = outpPath;
    if (!outpPathTrimmed.empty() && outpPathTrimmed.back() == '/') {
        outpPathTrimmed.pop_back(); // remove trailing slash
    }
    size_t lastSlash = outpPathTrimmed.find_last_of('/');
    if (lastSlash != std::string::npos) {
        subcktPath = outpPathTrimmed.substr(0, lastSlash) + "/ACT_out";
    } 
    else
        assert(0);
    cout << "subcktPath: " << subcktPath << endl;

    // [initialize oriId]
    Abc_Obj_t * pObj;
    ll i;
    Abc_NtkForEachObj(currNet.GetNet(), pObj, i) {
        pObj->oriId = pObj->Id;
    }

    // [read partition_map.txt]
    // partition_map.txt is in the parent directory of outpPath
    std::string partitionMapPath;
    if (lastSlash != std::string::npos) {
        partitionMapPath = outpPathTrimmed.substr(0, lastSlash) + "/partition_map.txt";
    } else {
        assert(0);
    }
    
    filesystem::path partitionMapFile(partitionMapPath);
    if (!filesystem::exists(partitionMapFile)) {
        cout << "Error: partition_map.txt does not exist: " << partitionMapPath << endl;
        assert(0);
    }
    
    // parse partition_map.txt and store in a map: subcircuit_id -> SubcktInfo
    std::unordered_map<ll, SubcktInfo> subcktInfoMap;
    std::ifstream fin(partitionMapFile);
    if (!fin.is_open()) {
        cout << "Error: cannot open partition_map.txt: " << partitionMapPath << endl;
        assert(0);
    }
    
    std::string line;
    ll currentSubId = -1;
    SubcktInfo currentInfo;
    while (std::getline(fin, line)) {
        if (line.empty()) {
            continue;
        }
        // check for "# Subcircuit n"
        if (line[0] == '#') {
            // skip comment line
            continue;
        }
        // check for "SUB_ID n"
        if (line.find("SUB_ID") == 0) {
            std::istringstream iss(line);
            std::string token;
            iss >> token; // "SUB_ID"
            iss >> currentSubId;
            currentInfo.id = currentSubId;
            currentInfo.LO_ids.clear();
            currentInfo.LI_ids.clear();
        }
        // check for "LO (count): id1 id2 ..."
        else if (line.find("LO (") == 0) {
            std::istringstream iss(line);
            std::string token;
            iss >> token; // "LO"
            iss >> token; // "(count):"
            ll id;
            while (iss >> id) {
                currentInfo.LO_ids.push_back(id);
            }
        }
        // check for "LI (count): id1 id2 ..."
        else if (line.find("LI (") == 0) {
            std::istringstream iss(line);
            std::string token;
            iss >> token; // "LI"
            iss >> token; // "(count):"
            ll id;
            while (iss >> id) {
                currentInfo.LI_ids.push_back(id);
            }
        }
        // check for "FILE filename.aig"
        else if (line.find("FILE") == 0) {
            std::istringstream iss(line);
            std::string token;
            iss >> token; // "FILE"
            iss >> currentInfo.filename;
            // store the completed SubcktInfo
            subcktInfoMap[currentSubId] = currentInfo;
        }
    }
    fin.close();
    
    cout << "Loaded " << subcktInfoMap.size() << " subcircuit infos from partition_map.txt" << endl;
    
    // [substitute each subcircuit]
    // traverse all files in subcktPath (format: xxx_sub_n.aig, n = 0, 1, 2...)
    filesystem::path subcktDir(subcktPath);
    if (!filesystem::exists(subcktDir)) {
        cout << "Error: subcktPath does not exist: " << subcktPath << endl;
        assert(0);
    }
    
    // regex pattern to match xxx_sub_n.aig files and extract subcircuit ID
    std::regex pattern(R"(.+_sub_(\d+)\.aig)");
    std::vector<std::pair<ll, std::string>> subcktFiles; // (subId, filePath)
    
    // iterate through directory and find matching files
    for (const auto & entry : filesystem::directory_iterator(subcktDir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            std::smatch match;
            if (std::regex_match(filename, match, pattern)) {
                ll subId = std::stoll(match[1].str());
                subcktFiles.push_back({subId, entry.path().string()});
            }
        }
    }
    
    // sort files by subcircuit ID for consistent processing
    std::sort(subcktFiles.begin(), subcktFiles.end(), 
              [](const std::pair<ll, std::string>& a, const std::pair<ll, std::string>& b) {
                  return a.first < b.first;
              });
    
    cout << "Found " << subcktFiles.size() << " subcircuit files" << endl;
    
    // read each subcircuit file using abc command
    Abc_Frame_t * pAbc = Abc_FrameGetGlobalFrame();
    for (const auto & filePair : subcktFiles) {
        ll subId = filePair.first;
        const std::string & filePath = filePair.second;
        
        // get LO and LI from partition_map.txt
        if (subcktInfoMap.find(subId) == subcktInfoMap.end()) {
            cout << "Error: subcircuit ID " << subId << " not found in partition_map.txt" << endl;
            assert(0);
        }
        const SubcktInfo & info = subcktInfoMap[subId];
        const std::vector<ll> & LO_ids = info.LO_ids;
        const std::vector<ll> & LI_ids = info.LI_ids;
        
        cout << "Reading subcircuit " << subId << ": " << filePath << endl;
        cout << "  LO (" << LO_ids.size() << "): ";
        for (ll id : LO_ids) cout << id << " ";
        cout << endl;
        cout << "  LI (" << LI_ids.size() << "): ";
        for (ll id : LI_ids) cout << id << " ";
        cout << endl;
        
        std::string cmd = "read " + filePath + "; logic; sop;";
        assert(!Cmd_CommandExecute(pAbc, cmd.c_str()));
        Abc_Ntk_t * pSubNtk = Abc_NtkDup(Abc_FrameReadNtk(pAbc));

        // ref:
        // Abc_FrameReplaceCurrentNetwork(pAbc, Abc_NtkDup(pSubNtk));
        // string Command = string("strash;");
        // assert(!Cmd_CommandExecute(pAbc, Command.c_str()));
        // Abc_Ntk_t * pSynNtk = Abc_NtkDup(Abc_FrameReadNtk(pAbc));
        
        // [do substitution using LO_ids and LI_ids]
        vector <ll> vLO, vLI;
        for (ll id : LO_ids) {
            vLO.push_back(currNet.GetNewId(id));
        }
        for (ll id : LI_ids) {
            vLI.push_back(currNet.GetNewId(id));
        }
        currNet.ReplaceSubCktPro(vLI, vLO, pSubNtk, LO_ids);
        
        auto netTmp = currNet;  // avoid inconsistent node Id
        netTmp.IsOriIdAll0("netTmp");
        currNet = netTmp;
        currNet.IsOriIdAll0("after duplicating");
        currNet.UpdateNewIdMap();
        cout << "oriId 12176 corresponds to newId: " << currNet.GetNewId(12176) << endl;
        // currNet.ReArrInTopoOrd();
        // cout << "finish topo" << endl;
        // currNet.UpdateNewIdMap();
        // cout << "oriId 12176 corresponds to newId: " << currNet.GetNewId(12176) << endl;

        // evaluation
        vector <ll> RealCom;
        for (auto i = 0; i < nOutput; ++i)
            RealCom.emplace_back(0);    // set compensation to 0
        assert(currNet.GetPoNum() % nOutput == 0);
        double err = CalcErr(accNet, currNet, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
        // double err = -1;
        cout << "after substituting subcircuit " << subId << ", " << metrType << " = " << err << ", size = " << currNet.GetArea() << ", depth = " << currNet.GetDelay() << endl; 
        // cout << "after substituting subcircuit " << subId << ", " << metrType << " = " << err << ", size = " << currNet.GetArea() << endl;     
        vErr.push_back(err);
    }

    // delete fake POs
    // cout << "realPoNum = " << realPoNum << ", currNet.GetPoNum() = " << currNet.GetPoNum() << endl;
    // for (ll i = realPoNum; i < currNet.GetPoNum(); i++) {
    //     Abc_NtkDeleteObjPo(currNet.GetPo(i));
    // }

    ExactSimpl(currNet, 0, 0);  // delete dangling nodes

    vector <ll> RealCom;
    for (auto i = 0; i < nOutput; ++i)
        RealCom.emplace_back(0);    // set compensation to 0
    assert(currNet.GetPoNum() % nOutput == 0);
    double err = CalcErr(accNet, currNet, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);   

    ostringstream oss("");
    oss << outpPath << accCktName << "_size_" << currNet.GetArea() << "_depth_" << currNet.GetDelay() << "_" << metrType << "_" << err;
    currNet.WriteNet(oss.str() + ".v", true); 
    
    // output vErr
    ostringstream errOss("");
    errOss << outpPath << accCktName << "_" << metrType << "_vErr.txt";
    std::ofstream errFile(errOss.str());
    if (errFile.is_open()) {
        for (const auto & errVal : vErr) {
            errFile << errVal << std::endl;
        }
        errFile.close();
        cout << "vErr output to: " << errOss.str() << endl;
    } else {
        cout << "Error: cannot open file for writing vErr: " << errOss.str() << endl;
    }
    
}

void ALSMan::GraphMerge_greedy() {
    cout << "Graph merge begin!" << endl;
    if (accNet.GetNetType() == NET_TYPE::SOP) {
        cout << "SOP network" << endl;
    }
    else if (accNet.GetNetType() == NET_TYPE::AIG) {
        cout << "AIG network" << endl;
    }
    else if (accNet.GetNetType() == NET_TYPE::STRASH) {
        cout << "STRASH network" << endl;
    }
    accNet.ConvToSop();
    auto currNet = accNet;
    ll realPoNum = Abc_NtkPoNum(currNet.GetNet());
    // accNet.PrintPro(1, 1, 0); 

    // [obtain subcktPath]
    // subcktPath is the ACT_out folder in the parent directory of outpPath
    // e.g., if outpPath is graph_merge/EPFL/random_control/arbiter/merge_out
    // then subcktPath is graph_merge/EPFL/random_control/arbiter/ACT_out
    // Note: outpPath may have trailing slash due to FixPath()
    std::string outpPathTrimmed = outpPath;
    if (!outpPathTrimmed.empty() && outpPathTrimmed.back() == '/') {
        outpPathTrimmed.pop_back(); // remove trailing slash
    }
    size_t lastSlash = outpPathTrimmed.find_last_of('/');
    if (lastSlash != std::string::npos) {
        subcktPath = outpPathTrimmed.substr(0, lastSlash) + "/ACT_out";
    } 
    else
        assert(0);
    cout << "subcktPath: " << subcktPath << endl;

    // [initialize oriId]
    Abc_Obj_t * pObj;
    ll i;
    Abc_NtkForEachObj(currNet.GetNet(), pObj, i) {
        pObj->oriId = pObj->Id;
    }

    // [read partition_map.txt]
    // partition_map.txt is in the parent directory of outpPath
    std::string partitionMapPath;
    if (lastSlash != std::string::npos) {
        partitionMapPath = outpPathTrimmed.substr(0, lastSlash) + "/partition_map.txt";
    } else {
        assert(0);
    }
    
    filesystem::path partitionMapFile(partitionMapPath);
    if (!filesystem::exists(partitionMapFile)) {
        cout << "Error: partition_map.txt does not exist: " << partitionMapPath << endl;
        assert(0);
    }
    
    // parse partition_map.txt and store in a map: subcircuit_id -> SubcktInfo
    std::unordered_map<ll, SubcktInfo> subcktInfoMap;
    std::ifstream fin(partitionMapFile);
    if (!fin.is_open()) {
        cout << "Error: cannot open partition_map.txt: " << partitionMapPath << endl;
        assert(0);
    }
    
    std::string line;
    ll currentSubId = -1;
    SubcktInfo currentInfo;
    while (std::getline(fin, line)) {
        if (line.empty()) {
            continue;
        }
        // check for "# Subcircuit n"
        if (line[0] == '#') {
            // skip comment line
            continue;
        }
        // check for "SUB_ID n"
        if (line.find("SUB_ID") == 0) {
            std::istringstream iss(line);
            std::string token;
            iss >> token; // "SUB_ID"
            iss >> currentSubId;
            currentInfo.id = currentSubId;
            currentInfo.LO_ids.clear();
            currentInfo.LI_ids.clear();
        }
        // check for "LO (count): id1 id2 ..."
        else if (line.find("LO (") == 0) {
            std::istringstream iss(line);
            std::string token;
            iss >> token; // "LO"
            iss >> token; // "(count):"
            ll id;
            while (iss >> id) {
                currentInfo.LO_ids.push_back(id);
            }
        }
        // check for "LI (count): id1 id2 ..."
        else if (line.find("LI (") == 0) {
            std::istringstream iss(line);
            std::string token;
            iss >> token; // "LI"
            iss >> token; // "(count):"
            ll id;
            while (iss >> id) {
                currentInfo.LI_ids.push_back(id);
            }
        }
        // check for "FILE filename.aig"
        else if (line.find("FILE") == 0) {
            std::istringstream iss(line);
            std::string token;
            iss >> token; // "FILE"
            iss >> currentInfo.filename;
            // store the completed SubcktInfo
            subcktInfoMap[currentSubId] = currentInfo;
        }
    }
    fin.close();
    
    cout << "Loaded " << subcktInfoMap.size() << " subcircuit infos from partition_map.txt" << endl;
    
    // [substitute each subcircuit]
    // traverse all files in subcktPath (format: xxx_sub_n.aig, n = 0, 1, 2...)
    filesystem::path subcktDir(subcktPath);
    if (!filesystem::exists(subcktDir)) {
        cout << "Error: subcktPath does not exist: " << subcktPath << endl;
        assert(0);
    }
    
    // regex pattern to match xxx_sub_n.aig files and extract subcircuit ID
    std::regex pattern(R"(.+_sub_(\d+)\.aig)");
    std::vector<std::pair<ll, std::string>> subcktFiles; // (subId, filePath)
    
    // iterate through directory and find matching files
    for (const auto & entry : filesystem::directory_iterator(subcktDir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            std::smatch match;
            if (std::regex_match(filename, match, pattern)) {
                ll subId = std::stoll(match[1].str());
                subcktFiles.push_back({subId, entry.path().string()});
            }
        }
    }
    
    // sort files by subcircuit ID for consistent processing
    std::sort(subcktFiles.begin(), subcktFiles.end(), 
              [](const std::pair<ll, std::string>& a, const std::pair<ll, std::string>& b) {
                  return a.first < b.first;
              });
    
    cout << "Found " << subcktFiles.size() << " subcircuit files" << endl;
    
    
    // greedy single-selection
    double currErr = 0;
    vector <ll> vApplied;
    ll nRound = 0;

    vector <ll> RealCom;
    for (auto i = 0; i < nOutput; ++i)
        RealCom.emplace_back(0);    // set compensation to 0
    assert(currNet.GetPoNum() % nOutput == 0);
    
    while (currErr < errUppBound) {
        cout << "============== round " << nRound << " begin ==============" << endl;
        seed = NewSeed();
        double backErr = CalcErr(accNet, currNet, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
        cout << "backErr = " << backErr << endl;
        if (backErr > errUppBound) {
            cout << "backErr > errUppBound, break" << endl;
            break;
        }
        vector <pair <ll, double>> vErr;
        for (ll i = 0; i < static_cast<ll>(subcktFiles.size()); ++i) {
            if (std::find(vApplied.begin(), vApplied.end(), i) != vApplied.end())
                continue; // already applied
            NetMan tmpNet = currNet;
            double err = ReplaceSubcircuit(i, subcktFiles, subcktInfoMap, tmpNet, false);
            vErr.push_back({i, err});
            if (err == 0)
                break;
        }
        std::sort(vErr.begin(), vErr.end(), 
                  [](const std::pair<ll, double>& a, const std::pair<ll, double>& b) {
                      return a.second < b.second;
                  });
        ll bestIndex = vErr[0].first;
        double bestErr = vErr[0].second;
        if (bestErr > errUppBound) {
            cout << "bestErr = " << bestErr << " > errUppBound, break" << endl;
            break;
        }

        // apply
        cout << "apply subcircuit " << bestIndex << ", err = " << bestErr << endl;
        ReplaceSubcircuit(bestIndex, subcktFiles, subcktInfoMap, currNet, true);
        // ExactSimpl(currNet, 0, 0);  // delete dangling nodes

        vector <ll> RealCom;
        for (auto i = 0; i < nOutput; ++i)
            RealCom.emplace_back(0);    // set compensation to 0
        assert(currNet.GetPoNum() % nOutput == 0);
        currErr = CalcErr(accNet, currNet, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);   

        vApplied.push_back(bestIndex);

        NetMan simplifiedNet = currNet;
        ExactSimpl(simplifiedNet, 0, 0);  // delete dangling nodes
        ostringstream oss("");
        oss << outpPath << nRound << "_" << accCktName << "_size_" << simplifiedNet.GetArea() << "_depth_" << simplifiedNet.GetDelay() << "_" << metrType << "_" << currErr;
        simplifiedNet.WriteNet(oss.str() + ".v", true);   

        ++nRound;
    }
    // ExactSimpl(currNet, 0, 0);  // delete dangling nodes
    // ostringstream oss("");
    // oss << outpPath << "final_" << accCktName << "_size_" << currNet.GetArea() << "_depth_" << currNet.GetDelay() << "_" << metrType << "_" << currErr;
    // currNet.WriteNet(oss.str() + ".v", true); 
}


void ALSMan::GraphMerge_binary() {
    auto startGraphMergeBinary = chrono::system_clock::now();
    cout << "Graph merge begin!" << endl;
    if (accNet.GetNetType() == NET_TYPE::SOP) {
        cout << "SOP network" << endl;
    }
    else if (accNet.GetNetType() == NET_TYPE::AIG) {
        cout << "AIG network" << endl;
    }
    else if (accNet.GetNetType() == NET_TYPE::STRASH) {
        cout << "STRASH network" << endl;
    }
    accNet.ConvToSop();
    auto currNet = accNet;
    ll realPoNum = Abc_NtkPoNum(currNet.GetNet());
    // accNet.PrintPro(1, 1, 0); 

    // [obtain subcktPath]
    // subcktPath is the ACT_out folder in the parent directory of outpPath
    // e.g., if outpPath is graph_merge/EPFL/random_control/arbiter/merge_out
    // then subcktPath is graph_merge/EPFL/random_control/arbiter/ACT_out
    // Note: outpPath may have trailing slash due to FixPath()
    std::string outpPathTrimmed = outpPath;
    if (!outpPathTrimmed.empty() && outpPathTrimmed.back() == '/') {
        outpPathTrimmed.pop_back(); // remove trailing slash
    }
    size_t lastSlash = outpPathTrimmed.find_last_of('/');
    if (lastSlash != std::string::npos) {
        subcktPath = outpPathTrimmed.substr(0, lastSlash) + "/ACT_out";
    } 
    else
        assert(0);
    cout << "subcktPath: " << subcktPath << endl;

    // [initialize oriId]
    Abc_Obj_t * pObj;
    ll i;
    Abc_NtkForEachObj(currNet.GetNet(), pObj, i) {
        pObj->oriId = pObj->Id;
    }

    // [read partition_map.txt]
    // partition_map.txt is in the parent directory of outpPath
    std::string partitionMapPath;
    if (lastSlash != std::string::npos) {
        partitionMapPath = outpPathTrimmed.substr(0, lastSlash) + "/partition_map.txt";
    } else {
        assert(0);
    }
    
    filesystem::path partitionMapFile(partitionMapPath);
    if (!filesystem::exists(partitionMapFile)) {
        cout << "Error: partition_map.txt does not exist: " << partitionMapPath << endl;
        assert(0);
    }
    
    // parse partition_map.txt and store in a map: subcircuit_id -> SubcktInfo
    std::unordered_map<ll, SubcktInfo> subcktInfoMap;
    std::ifstream fin(partitionMapFile);
    if (!fin.is_open()) {
        cout << "Error: cannot open partition_map.txt: " << partitionMapPath << endl;
        assert(0);
    }
    
    std::string line;
    ll currentSubId = -1;
    SubcktInfo currentInfo;
    while (std::getline(fin, line)) {
        if (line.empty()) {
            continue;
        }
        // check for "# Subcircuit n"
        if (line[0] == '#') {
            // skip comment line
            continue;
        }
        // check for "SUB_ID n"
        if (line.find("SUB_ID") == 0) {
            std::istringstream iss(line);
            std::string token;
            iss >> token; // "SUB_ID"
            iss >> currentSubId;
            currentInfo.id = currentSubId;
            currentInfo.LO_ids.clear();
            currentInfo.LI_ids.clear();
        }
        // check for "LO (count): id1 id2 ..."
        else if (line.find("LO (") == 0) {
            std::istringstream iss(line);
            std::string token;
            iss >> token; // "LO"
            iss >> token; // "(count):"
            ll id;
            while (iss >> id) {
                currentInfo.LO_ids.push_back(id);
            }
        }
        // check for "LI (count): id1 id2 ..."
        else if (line.find("LI (") == 0) {
            std::istringstream iss(line);
            std::string token;
            iss >> token; // "LI"
            iss >> token; // "(count):"
            ll id;
            while (iss >> id) {
                currentInfo.LI_ids.push_back(id);
            }
        }
        // check for "FILE filename.aig"
        else if (line.find("FILE") == 0) {
            std::istringstream iss(line);
            std::string token;
            iss >> token; // "FILE"
            iss >> currentInfo.filename;
            // store the completed SubcktInfo
            subcktInfoMap[currentSubId] = currentInfo;
        }
    }
    fin.close();
    
    cout << "Loaded " << subcktInfoMap.size() << " subcircuit infos from partition_map.txt" << endl;
    
    // [substitute each subcircuit]
    // traverse all files in subcktPath (format: xxx_sub_n.aig, n = 0, 1, 2...)
    filesystem::path subcktDir(subcktPath);
    if (!filesystem::exists(subcktDir)) {
        cout << "Error: subcktPath does not exist: " << subcktPath << endl;
        assert(0);
    }
    
    // regex pattern to match xxx_sub_n.aig files and extract subcircuit ID
    std::regex pattern(R"(.+_sub_(\d+)\.aig)");
    std::vector<std::pair<ll, std::string>> subcktFiles; // (subId, filePath)
    
    // iterate through directory and find matching files
    for (const auto & entry : filesystem::directory_iterator(subcktDir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            std::smatch match;
            if (std::regex_match(filename, match, pattern)) {
                ll subId = std::stoll(match[1].str());
                subcktFiles.push_back({subId, entry.path().string()});
            }
        }
    }
    
    // sort files by subcircuit ID for consistent processing
    std::sort(subcktFiles.begin(), subcktFiles.end(), 
              [](const std::pair<ll, std::string>& a, const std::pair<ll, std::string>& b) {
                  return a.first < b.first;
              });
    
    cout << "Found " << subcktFiles.size() << " subcircuit files" << endl;
    
    
    // binary-selection
    double currErr = 0;
    vector <ll> vApplied;
    ll nRound = 0;

    vector <ll> RealCom;
    for (auto i = 0; i < nOutput; ++i)
        RealCom.emplace_back(0);    // set compensation to 0
    assert(currNet.GetPoNum() % nOutput == 0);

    while (currErr <= errUppBound) {
        cout << "============== round " << nRound << " begin ==============" << endl;
        // update seed
        seed = NewSeed();
        double backErr = CalcErr(accNet, currNet, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
        cout << "backErr = " << backErr << endl;
        if (backErr > errUppBound) {
            cout << "backErr > errUppBound, break" << endl;
            break;
        }
        vector <pair <ll, double>> vErr;
        vector <pair <ll, double>> vErr_exceed_bound;
        for (ll i = 0; i < static_cast<ll>(subcktFiles.size()); ++i) {
            if (std::find(vApplied.begin(), vApplied.end(), i) != vApplied.end())
                continue; // already applied
            NetMan tmpNet = currNet;
            ReplaceSubcircuit_v2(i, subcktFiles, subcktInfoMap, tmpNet, false);
            double err = CalcErr(accNet, tmpNet, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
            if (err <= errUppBound)
                vErr.push_back({i, err});
            else
                vErr_exceed_bound.push_back({i, err});
        }
        if (vErr.empty()) {
            cout << "vErr is empty, break" << endl;
            for (auto it = vErr_exceed_bound.begin(); it != vErr_exceed_bound.end(); ++it) {
                cout << "subcircuit " << it->first << " err = " << it->second << " > errUppBound" << endl;
            }
            break;
        }
        std::sort(vErr.begin(), vErr.end(), 
                  [](const std::pair<ll, double>& a, const std::pair<ll, double>& b) {
                      return a.second < b.second;
                  });
        bool fContinue = true;
        while (fContinue) {
            cout << "vErr size = " << vErr.size() << endl;
            if (vErr.empty()) {
                cout << "after binary pruning, vErr is empty, break" << endl;
                currErr = errUppBound * 2;
                break;
            }
            vector <ll> vCand;
            if (vErr.size() > 1) {
                for (auto it = vErr.begin(); it != vErr.begin() + vErr.size() / 2; ++it) {
                    vCand.push_back(it->first);
                }
            }
            else {
                vCand.push_back(vErr[0].first);
            }
            // apply
            NetMan tmpNet = currNet;
            for (ll i = 0; i < vCand.size(); ++i) {
                ReplaceSubcircuit_v2(vCand[i], subcktFiles, subcktInfoMap, tmpNet, false);
            }
            currErr = CalcErr(accNet, tmpNet, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
            cout << "after applying subcircuits {";
            for (ll i = 0; i < vCand.size(); ++i) {
                cout << vCand[i] << ", ";
            }
            cout << "}, " << endl;
            if (currErr <= errUppBound) {
                cout << "err = " << currErr << endl;
                currNet = tmpNet;
                for (ll i = 0; i < vCand.size(); ++i) {
                    vApplied.push_back(vCand[i]);
                }
                fContinue = false;
            }
            else {
                cout << "err = " << currErr << " > errUppBound, continue pruning" << endl;
                if (vErr.size() > 1) {
                    vErr.resize(vErr.size() / 2);
                }
                else {  // only one subcircuit left, and it is applied already
                    vErr.clear();
                }
            }
        }

        if (currErr <= errUppBound) {
            NetMan simplifiedNet = currNet;
            ExactSimpl(simplifiedNet, 0, 0);  // delete dangling nodes
            double simplifiedErr = CalcErr(accNet, simplifiedNet, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
            ostringstream oss("");
            oss << outpPath << nRound << "_" << accCktName << "_size_" << simplifiedNet.GetArea() << "_depth_" << simplifiedNet.GetDelay() << "_" << metrType << "_" << simplifiedErr;
            simplifiedNet.WriteNet(oss.str() + ".v", true);   

            NetMan mapNet = simplifiedNet;
            // mapNet.Comm("amap;", true);
            // mapNet.Comm("st; compress2rs; ps; dch; amap;", true);  // st: SOP->AIG; amap expects AIG
            // Comm uses ABC global frame: set mapNet current, run, then read back result
            Abc_Frame_t * pAbc = Abc_FrameGetGlobalFrame();
            Abc_FrameReplaceCurrentNetwork(pAbc, Abc_NtkDup(mapNet.GetNet()));
            cout << "run amap;" << endl;
            assert(!Cmd_CommandExecute(pAbc, "amap;"));
            mapNet = NetMan(Abc_NtkDup(Abc_FrameReadNtk(pAbc)), true);
            mapNet.ReArrInTopoOrd();
            ostringstream oss2("");
            oss2 << outpPath << nRound << "_" << accCktName << "_area_" << mapNet.GetArea() << "_delay_" << mapNet.GetDelay() << "_" << metrType << "_" << simplifiedErr;
            mapNet.WriteNet(oss2.str() + ".v", true);
        }

        ++nRound;
    }
    auto durationGraphMergeBinary = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - startGraphMergeBinary);
    cout << "runtime for GraphMerge_binary = " << double(durationGraphMergeBinary.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " sec" << endl;
}


void ALSMan::GraphPartition() {
    // if (accNet.GetNetType() == NET_TYPE::SOP) {
    //     cout << "SOP network" << endl;
    // }
    // else if (accNet.GetNetType() == NET_TYPE::AIG) {
    //     cout << "AIG network" << endl;
    // }
    // else if (accNet.GetNetType() == NET_TYPE::STRASH) {
    //     cout << "STRASH network" << endl;
    // }
    assert(accNet.GetNetType() == NET_TYPE::STRASH);    // read .aig file
    accNet.ConvToSop();
    accNet.ReArrInTopoOrd();
    // accNet.PrintPro(1, 1, 0); 
    // cout << endl;  
    // output the original network in sop format
    ostringstream oss("");
    oss << outpPath << "acc_sop_" << accCktName << "_size_" << accNet.GetArea() << "_depth_" << accNet.GetDelay();
    accNet.WriteNet(oss.str() + ".v", true);
    
    // determine LO pairs; extract and output subcircuits
    GetSubcktsPro(accNet);
    // output subckt infos
    PrintSubcktInfos();

    cout << "Graph partition finished" << endl;
}

void ALSMan::GetSubcktsPro(NetMan & net) {
    // use fMarkA to mark the subcircuits that have been extracted (internal nodes and LOs) (don't-touch).
    Abc_NtkCleanMarkA(net.GetNet());

    // parameter
    ll maxNodeNum = 50;
    ll maxNodeNumWithPat = 75;  // with patience

    bool fDebug = false;

    // main loop
    ll subcktId = 0;
    while (!net.IsAllNodeMarkA1()) { // extract one subgraph in one iteration
        // use fMarkB to mark the nodes that are in the subcircuit
        // LI: fMarkB = 0; fMarkC = 1
        // LO: fMarkB = 1; exist a fanout whose fMarkB = 0
        // Internal nodes: fMarkB = 1; for all of its fanouts: fMarkB = 1
        Abc_NtkCleanMarkB(net.GetNet());
        Abc_NtkCleanMarkC(net.GetNet());
        Abc_Obj_t * pNode;
        int i;
        ll startId = 0;
        ll remainingNodeNum = 0;
        Abc_NtkForEachNode(net.GetNet(), pNode, i) {
            if (pNode->fMarkA)
                continue;
            ++remainingNodeNum;
            if (startId == 0) {
                startId = pNode->Id;
                pNode->fMarkB = 1;
            }
        }
        ll nodeNumLim = maxNodeNum;
        if (remainingNodeNum > maxNodeNum && remainingNodeNum < maxNodeNumWithPat)
            nodeNumLim = maxNodeNumWithPat / 2;
        cout << "remainingNodeNum (fMarkA == 0) = " << remainingNodeNum << ", nodeNumLim = " << nodeNumLim << endl;
        cout << "startId = " << startId << " (" << Abc_ObjName(net.GetObj(startId)) << ")" << endl;
        
        // obtain a subcircuit
        set <ll> LOs = {startId};
        set <ll> LIs;
        for (ll i = 0; i < net.GetFaninNum(startId); ++i) {
            Abc_Obj_t * pFanin = net.GetFanin(startId, i);
            LIs.insert(pFanin->Id);
            pFanin->fMarkC = 1;
        }
        ll nodeNum = 1;

        do {
            // back up
            auto backUpNet = net;
            auto backUpLOs = LOs;
            auto backUpLIs = LIs;
            auto backUpnodeNum = nodeNum;

            // select a fanout of an LO to add into the subcircuit (and update LI)
            ll expandId = 0;
            Abc_Obj_t * pExpandNode;
            set <ll> fanouts;
            // consider a fanout as the expand node
            for (auto it = LOs.begin(); it != LOs.end(); ++it) {
                Abc_Obj_t * pLO = net.GetObj(*it);
                for (ll i = 0; i < net.GetFanoutNum(pLO); ++i) {
                    fanouts.insert(net.GetFanoutId(pLO, i));
                }
            }
            for (auto it = fanouts.begin(); it != fanouts.end(); ++it) {  // topo order
                ll fanoutId = *it;
                Abc_Obj_t * pFanout = net.GetObj(fanoutId);
                if (pFanout->fMarkA || pFanout->fMarkB)
                    continue;
                if (Abc_ObjIsPo(pFanout))
                    continue;
                expandId = fanoutId;
                pExpandNode = pFanout;
                break;
            }
            // if no fanout can be expanded, consider other nodes
            if (expandId == 0) {
                // if (fDebug) {
                //     cout << "expandId = 0" << endl;
                //     cout << "LOs = {";
                //     for (auto it = LOs.begin(); it != LOs.end(); ++it)
                //         cout << Abc_ObjName(net.GetObj(*it)) << " ";
                //     cout << "}" << endl;
                // }
                Abc_NtkForEachNode(net.GetNet(), pNode, i) {
                    if (pNode->fMarkA)
                        continue;
                    if (pNode->fMarkB)
                        continue;
                    expandId = pNode->Id;
                    pExpandNode = pNode;
                    break;
                }
                if (expandId == 0) {
                    if (fDebug) {
                        cout << "expandId = 0" << endl;
                    }
                    break;
                }
            }

            // expand
            if (fDebug)
                cout << "expand " << Abc_ObjName(pExpandNode) << endl;
            pExpandNode->fMarkB = 1;
            pExpandNode->fMarkC = 0;
            // update LI
            // Abc_NtkCleanMarkC(net.GetNet());
            for (ll i = 0; i < net.GetFaninNum(expandId); ++i) {
                Abc_Obj_t * pFanin = net.GetFanin(expandId, i);
                if (!pFanin->fMarkB && !pFanin->fMarkC) {
                    LIs.insert(pFanin->Id);
                    pFanin->fMarkC = 1;
                }
            }
            if (fDebug) {
                cout << "LIs = {";
                for (auto it = LIs.begin(); it != LIs.end(); ++it)
                    cout << Abc_ObjName(net.GetObj(*it)) << " ";
                cout << "}" << endl;
            }
            // remove redundant LIs
            for (auto it = LIs.begin(); it != LIs.end(); ) {
                if (net.IsObjPi(*it)) { 
                    ++it;
                    continue;
                }
                if (net.GetObj(*it)->fMarkA) {
                    ++it;
                    continue;
                }
                bool fRedundant = true;
                for (ll i = 0; i < net.GetFaninNum(*it); ++i) {
                    if (!net.GetFanin(*it, i)->fMarkC) {
                        fRedundant = false;
                        ++it;
                        break;
                    }
                }
                if (fRedundant) {
                    // not LI any more, but an internal node
                    net.GetObj(*it)->fMarkC = 0;
                    net.GetObj(*it)->fMarkB = 1;
                    LIs.erase(it++);
                }
            }
            if (fDebug) {
                cout << "remove redundant LIs: LIs = {";
                for (auto it = LIs.begin(); it != LIs.end(); ++it)
                    cout << Abc_ObjName(net.GetObj(*it)) << " ";
                cout << "}" << endl;
            }
            // update internal nodes & LOs
            nodeNum = 0;
            Abc_NtkForEachNode(net.GetNet(), pNode, i) {
                if (pNode->Id <= *(LIs.begin()))
                    continue;
                if (pNode->fMarkC || pNode->fMarkA)
                    continue;
                if (pNode->fMarkB) {
                    ++nodeNum;
                    continue;
                }
                ll j;
                Abc_Obj_t * pFanin;
                bool IsInSubGraph = true;
                Abc_ObjForEachFanin(pNode, pFanin, j) {
                    if (!pFanin->fMarkB && !pFanin->fMarkC) {
                        IsInSubGraph = false;
                        break;
                    }
                }
                if (IsInSubGraph) {
                    pNode->fMarkB = 1;
                    ++nodeNum;
                }
            }
            if (fDebug) {
                cout << "nodeNum = " << nodeNum << endl;
            }
            if (nodeNum > nodeNumLim) {
                if (fDebug) {
                    cout << "nodeNum = " << nodeNum << ", nodeNumLim = " << nodeNumLim << ". nodeNum > nodeNumLim, back up and break" << endl;
                }
                net = backUpNet;
                LOs = backUpLOs;
                LIs = backUpLIs;
                nodeNum = backUpnodeNum;
                break;
            }
            // obtain LOs
            LOs.clear();
            Abc_NtkForEachNode(net.GetNet(), pNode, i) {
                if (!pNode->fMarkB) 
                    continue;
                Abc_Obj_t * pFanout;
                ll j;
                Abc_ObjForEachFanout(pNode, pFanout, j) {
                    if (!pFanout->fMarkB) {
                        LOs.insert(pNode->Id);
                        break;
                    }
                }
            }
        } while (nodeNum <= nodeNumLim);

        // check nodeNum
        ll sum = 0;
        Abc_NtkForEachNode(net.GetNet(), pNode, i) {
            if (pNode->fMarkB)
                ++sum;
        }
        if (sum != nodeNum) {
            cout << "nodeNum = " << nodeNum << ", sum = " << sum << endl;
            cout << "nodeNum is not correct" << endl;
            cout << "LIs: " << endl;
            for (const auto & li_id : LIs) {
                cout << Abc_ObjName(net.GetObj(li_id)) << ", fMarkA = " << net.GetObj(li_id)->fMarkA << ", fMarkB = " << net.GetObj(li_id)->fMarkB << ", fMarkC = " << net.GetObj(li_id)->fMarkC << endl;
            }
            cout << endl << "LOs: " << endl;
            for (const auto & lo_id : LOs) {
                cout << Abc_ObjName(net.GetObj(lo_id)) << ", fMarkA = " << net.GetObj(lo_id)->fMarkA << ", fMarkB = " << net.GetObj(lo_id)->fMarkB << ", fMarkC = " << net.GetObj(lo_id)->fMarkC << endl;
            }
            cout << endl;
            
            assert(sum == nodeNum);
        }

        // output the subcircuit
        if (fDebug) {
            cout << "output the subcircuit" << endl;
        }
        vector <ll> vLO(LOs.begin(), LOs.end());
        vector <ll> vLI(LIs.begin(), LIs.end());
        OutputCurrSubckt(net, vLO, vLI, subcktId, nodeNum);
        ++subcktId;
    }
}

void ALSMan::OutputCurrSubckt(NetMan & net, std::vector <ll> & vLO, std::vector <ll> & vLI, ll subcktId, ll nodeNum) {
    bool fDebug = false;
    // duplicate the nodes
    if (fDebug)
        cout << "duplicate the nodes" << endl;
    ll i;
    Abc_Obj_t * pNode, * pFanin;
    Abc_Ntk_t * pSubNtk = Abc_NtkAlloc(ABC_NTK_LOGIC, ABC_FUNC_SOP, 1);
    for (const auto & li : vLI) 
        Abc_NtkCreatePi(pSubNtk);
    for (const auto & lo : vLO) 
        Abc_NtkCreatePo(pSubNtk);
    Abc_NtkCleanCopy(net.GetNet());
    Abc_NtkForEachNodeReverse(net.GetNet(), pNode, i) {
        if (!pNode->fMarkB)
            continue;
        pNode->fMarkA = 1;
        Abc_NtkDupObj(pSubNtk, pNode, 0);  // copy node to pSubNtk
    }
    // connect all objects
    if (fDebug)
        cout << "connect the nodes" << endl;
    ll loIndex = 0;
    for (const auto & lo : vLO) {
        assert(net.GetObj(lo)->pCopy != nullptr);
        Abc_ObjAddFanin(Abc_NtkPo(pSubNtk, loIndex), net.GetObj(lo)->pCopy);
        ++loIndex;
    }
    Abc_NtkForEachNodeReverse(net.GetNet(), pNode, i) {
        if (!pNode->fMarkB)
            continue;
        assert(pNode->pCopy != nullptr);
        if (fDebug)
            cout << Abc_ObjName(pNode) << " ( fMarkB = " << pNode->fMarkB << ")" << endl;
        ll j;
        Abc_ObjForEachFanin(pNode, pFanin, j) {
            bool IsLI = false;
            for (ll k = 0; k < vLI.size(); k++) {
                if (pFanin->Id == vLI[k]) {
                    IsLI = true;
                    assert(k < Abc_NtkPiNum(pSubNtk));
                    Abc_ObjAddFanin(pNode->pCopy, Abc_NtkPi(pSubNtk, k));
                    break;
                }
            }
            if (!IsLI) {
                if (pFanin->pCopy == nullptr) {
                    cout << "pFanin->Id = " << pFanin->Id << endl;
                }
                assert(pFanin->pCopy != nullptr);
                Abc_ObjAddFanin(pNode->pCopy, pFanin->pCopy);
            }
        }
    }

    // transform to strash (.aig file)
    if (fDebug)
        cout << "transform to strash" << endl;
    Abc_Frame_t * pAbc = Abc_FrameGetGlobalFrame();
    Abc_FrameReplaceCurrentNetwork(pAbc, Abc_NtkDup(pSubNtk));
    string Command = string("strash;");
    assert(!Cmd_CommandExecute(pAbc, Command.c_str()));
    Abc_NtkDelete(pSubNtk);
    Abc_Ntk_t * pAigNtk = Abc_NtkDup(Abc_FrameReadNtk(pAbc));

    // output
    filesystem::path subDir = filesystem::path(outpPath) / "subCkts";
    if (!filesystem::exists(subDir)) {
        filesystem::create_directories(subDir); 
    }
    std::ostringstream oss;
    oss << accCktName << "_sub_" << subcktId;
    filesystem::path outAigFile = subDir / oss.str();
    NetMan subNet(pAigNtk, true);
    subNet.WriteNet(outAigFile.string() + ".aig", true);
    subNet.WriteNet(outAigFile.string() + ".blif", true);

    // save to vSubcktInfos
    vSubcktInfos.push_back({subcktId, vLO, vLI, nodeNum, {}, oss.str() + ".aig"});
    cout << "subcktId = " << subcktId << ": " << endl;
    cout << "LO (" << vLO.size() << "): ";
    for (int id : vLO)
        cout << " " << id;   
    cout << endl << "LI (" << vLI.size() << "): ";
    for (int id : vLI) 
        cout << " " << id;
    cout << endl << "NODE_NUM = " << nodeNum << endl << endl;
}

void ALSMan::GetSubckts_2LO(NetMan & net) {
    // use fMarkA to mark the subcircuits that have been extracted (internal nodes and LOs).
    Abc_NtkCleanMarkA(net.GetNet());
    Abc_Obj_t * pNode;
    int i;
    ll subcktId = 0;
    Abc_NtkForEachNodeReverse(net.GetNet(), pNode, i) {
        if (Abc_NodeIsConst(pNode))
            continue;
        if (Abc_ObjIsPi(Abc_ObjFanin0(pNode)))
            continue;
        if (pNode->fMarkA)
            continue;
        
        assert(Abc_ObjFaninNum(pNode) == 2);
        ll pairId = FindPairLO(net, pNode->Id);   // find the pair LO
        if (pairId == -1) {
            cout << "no pair LO found for " << pNode->Id << endl;
            continue;
        }
        vLos.push_back({pNode->Id, pairId});
        pNode->fMarkA = 1;
        net.GetObj(pairId)->fMarkA = 1;
        
        ExtractSubckt(net, vLos.back(), subcktId);    // extract and output the subcircuit
        ++subcktId;
    }

    Abc_NtkCleanMarkA(net.GetNet());
}

ll ALSMan::FindPairLO(NetMan & net, ll id) {
    ll TFI_Lev = 5;  // parameter: the level of the TFI cone considered
    set <ll> TFIs1 = net.GetPartialTFI(net.GetObj(id), TFI_Lev);
    if (TFIs1.empty())
        return -1;
    auto pNode = net.GetObj(id);
    assert(pNode != nullptr);
    assert(Abc_ObjIsNode(pNode));
    assert(!Abc_NodeIsConst(pNode));
    assert(Abc_ObjFaninNum(pNode) == 2);
    set <ll> fanouts;
    for (ll i = 0; i < net.GetFaninNum(id); ++i) {
        Abc_Obj_t * pFanin = net.GetFanin(id, i);
        for (ll j = 0; j < net.GetFanoutNum(pFanin); ++j) {
            if (net.GetFanoutId(pFanin, j) == id)
                continue;
            if (net.GetFanout(pFanin, j)->fMarkA)
                continue;
            fanouts.insert(net.GetFanoutId(pFanin, j));
        }
    }
    if (fanouts.empty()) {  // select a fanin
        auto pFanin0 = net.GetFanin(id, 0);
        auto pFanin1 = net.GetFanin(id, 1);
        if (pFanin0->fMarkA && pFanin1->fMarkA)
            return -1;
        else if (pFanin0->fMarkA)
            return pFanin1->Id;
        else if (pFanin1->fMarkA)
            return pFanin0->Id;
        else {
            return pFanin1->Id;  // temporary selection
        }
    }
    else {  // select an input-sharing node
        ll maxIntersecNum = 0;
        ll bestPairNodeId = -1;
        for (const auto & pairNodeId: fanouts) {
            set <ll> TFIs2 = net.GetPartialTFI(net.GetObj(pairNodeId), TFI_Lev);
            if (TFIs2.empty())
                continue;
            if (bestPairNodeId != -1)
                bestPairNodeId = pairNodeId;
            ll intersecNum = CountIntersection(TFIs1, TFIs2);
            if (intersecNum > maxIntersecNum) {
                maxIntersecNum = intersecNum;
                bestPairNodeId = pairNodeId;
            }
        }
        return bestPairNodeId;
    }
}

void ALSMan::ExtractSubckt(NetMan & net, std::vector <ll> & vLO, ll subcktId) {
    ll inpNumLim = 8;  // parameter: the number of LI in the subcircuit
    Abc_NtkCleanMarkB(net.GetNet());
    Abc_NtkCleanMarkC(net.GetNet());

    assert(vLO.size() == 2);
    bool fDebug = true;
    if (fDebug) 
        cout << "begin ExtractSubckt: vLO = {" << Abc_ObjName(net.GetObj(vLO[0])) << ", " << Abc_ObjName(net.GetObj(vLO[1])) << "}" << endl;
    
    // fMarkB = 1: nodes in the subcircuit; fMarkC = 1: have been expanded/explored
    auto pObj1 = net.GetObj(vLO[0]);
    assert(pObj1 != nullptr);
    auto pObj2 = net.GetObj(vLO[1]);
    assert(pObj2 != nullptr);
    pObj1->fMarkB = 1;
    pObj2->fMarkB = 1;
    set <ll> LIs;
    Abc_Obj_t * pFanin;
    ll k;
    Abc_ObjForEachFanin(pObj1, pFanin, k) {
        if (pFanin->Id != vLO[0] && pFanin->Id != vLO[1])
            LIs.insert(pFanin->Id);
    }
    Abc_ObjForEachFanin(pObj2, pFanin, k) {
        if (pFanin->Id != vLO[0] && pFanin->Id != vLO[1])
            LIs.insert(pFanin->Id);
    }
    vector <ll> vLI;
    vLI.assign(LIs.begin(), LIs.end()); 
    ll nodeNum = 2;
    while (1) {
        Abc_Obj_t * pExpandNode;
        ll expandId = 0;
        // select a node to expand forward (the direction to PI)
        for (auto it = LIs.rbegin(); it != LIs.rend(); ++it) {  // topo order (from largest ID to smallest ID)
            expandId = *it;
            pExpandNode = net.GetObj(expandId);
            if (!pExpandNode->fMarkC)
                break;
            else
                expandId = 0;
        }
        if (expandId == 0)
            break;

        // expand
        if (Abc_ObjIsPi(pExpandNode) || pExpandNode->fMarkA) {
            pExpandNode->fMarkC = 1;
            continue;
        }
        else {
            bool fCanExpand = true;
            Abc_Obj_t * pFanout;
            Abc_ObjForEachFanout(pExpandNode, pFanout, k) {
                if (!pFanout->fMarkB) {
                    fCanExpand = false;
                    break;
                }
            }

            pExpandNode->fMarkC = 1;    // mark: have been explored
            if (fCanExpand) {
                // mark as in MFFC
                bool fMarkBbackup = pExpandNode->fMarkB;
                pExpandNode->fMarkB = 1;
                // remove the expanded node
                auto it = LIs.find(expandId);
                assert(it != LIs.end());
                LIs.erase(it);
                // add the fanins of the expanded node
                Abc_ObjForEachFanin(pExpandNode, pFanin, k) {
                    LIs.insert(pFanin->Id);
                }

                // check FFW
                if (LIs.size() <= inpNumLim) {
                    if (fDebug) {
                        cout << "LI frontier: ";
                        for (const auto & li : LIs) {
                            cout << Abc_ObjName(net.GetObj(li)) << " ";
                        }
                    }
                    ll nMffc = 0;
                    Abc_Obj_t * pObj;
                    if (fDebug) 
                        cout << ". fMarkB nodes: ";
                    Abc_NtkForEachNode(net.GetNet(), pObj, k) {
                        if (pObj->fMarkB) {
                            ++nMffc;
                            if (fDebug) 
                                cout << Abc_ObjName(pObj) << " ";
                        }
                    }
                    if (fDebug) 
                        cout << endl;
                    if (nMffc >= 2) {
                        vLI.assign(LIs.begin(), LIs.end()); 
                        nodeNum = nMffc;                              
                    }
                }
                else 
                    pExpandNode->fMarkB = fMarkBbackup;
            }
        }
    }
    assert(nodeNum > 0);
    assert(vLI.size() != 0);

    // output the subcircuit
    // duplicate the nodes
    if (fDebug)
        cout << "duplicate the nodes" << endl;
    ll i;
    Abc_Obj_t * pNode;
    Abc_Ntk_t * pSubNtk = Abc_NtkAlloc(ABC_NTK_LOGIC, ABC_FUNC_SOP, 1);
    for (const auto & li : vLI) {
        Abc_NtkCreatePi(pSubNtk);
    }
    Abc_NtkCleanCopy(net.GetNet());
    vector <ll> vNodeIds;
    Abc_NtkForEachNodeReverse(net.GetNet(), pNode, i) {
        if (!pNode->fMarkB)
            continue;
        pNode->fMarkA = 1;
        if (pNode->Id != vLO[0] && pNode->Id != vLO[1])
            vNodeIds.push_back(pNode->Id);
        Abc_NtkDupObj(pSubNtk, pNode, 0);  // copy node to pSubNtk
    }
    // connect all objects
    if (fDebug)
        cout << "connect the nodes" << endl;
    auto pPo0 = Abc_NtkCreatePo(pSubNtk);
    Abc_ObjAddFanin(pPo0, net.GetObj(vLO[0])->pCopy);
    auto pPo1 = Abc_NtkCreatePo(pSubNtk);
    Abc_ObjAddFanin(pPo1, net.GetObj(vLO[1])->pCopy);
    Abc_NtkForEachNodeReverse(net.GetNet(), pNode, i) {
        if (!pNode->fMarkB)
            continue;
        assert(pNode->pCopy != nullptr);
        if (fDebug)
            cout << Abc_ObjName(pNode) << " ( fMarkB = " << pNode->fMarkB << ")" << endl;
        ll j;
        Abc_ObjForEachFanin(pNode, pFanin, j) {
            bool IsLI = false;
            for (ll k = 0; k < vLI.size(); k++) {
                if (pFanin->Id == vLI[k]) {
                    IsLI = true;
                    assert(k < Abc_NtkPiNum(pSubNtk));
                    Abc_ObjAddFanin(pNode->pCopy, Abc_NtkPi(pSubNtk, k));
                    break;
                }
            }
            if (!IsLI) {
                if (pFanin->pCopy == nullptr) {
                    cout << "pFanin->Id = " << pFanin->Id << endl;
                }
                assert(pFanin->pCopy != nullptr);
                Abc_ObjAddFanin(pNode->pCopy, pFanin->pCopy);
            }
        }
    }

    // transform to strash (.aig file)
    if (fDebug)
        cout << "transform to strash" << endl;
    Abc_Frame_t * pAbc = Abc_FrameGetGlobalFrame();
    Abc_FrameReplaceCurrentNetwork(pAbc, Abc_NtkDup(pSubNtk));
    string Command = string("strash;");
    assert(!Cmd_CommandExecute(pAbc, Command.c_str()));
    Abc_NtkDelete(pSubNtk);
    Abc_Ntk_t * pAigNtk = Abc_NtkDup(Abc_FrameReadNtk(pAbc));

    // output
    filesystem::path subDir = filesystem::path(outpPath) / "subCkts";
    if (!filesystem::exists(subDir)) {
        filesystem::create_directories(subDir); 
    }
    std::ostringstream oss;
    oss << accCktName << "_sub_" << subcktId;
    filesystem::path outAigFile = subDir / oss.str();
    NetMan subNet(pAigNtk, true);
    subNet.WriteNet(outAigFile.string() + ".aig", true);
    subNet.WriteNet(outAigFile.string() + ".blif", true);

    // save to vSubcktInfos
    vSubcktInfos.push_back({subcktId, vLO, vLI, -1, vNodeIds, oss.str() + ".aig"});
}

void ALSMan::PrintSubcktInfos() {
    filesystem::path dir(outpPath);
    if (!filesystem::exists(dir)) {
        filesystem::create_directories(dir);
    }
    filesystem::path filePath = dir / "partition_map.txt";

    std::ofstream fout(filePath);
    if (!fout.is_open()) {
        std::cerr << "Error: cannot open file for writing: " << filePath << std::endl;
        return;
    }
    ll i = 0;
    for (const auto& s : vSubcktInfos) {
        assert(i == s.id);
        ++i;

        fout << "# Subcircuit " << s.id << "\n";
        fout << "SUB_ID " << s.id << "\n";

        fout << "LO (" << s.LO_ids.size() << "): ";
        for (int id : s.LO_ids)
            fout << " " << id;
        
        fout << "\nLI (" << s.LI_ids.size() << "): ";
        for (int id : s.LI_ids) 
            fout << " " << id;
        
        // fout << "\nNODE (" << s.node_ids.size() << "): ";
        // for (int id : s.node_ids) 
        //     fout << " " << id;

        fout << "\nNODE_NUM = " << s.nodeNum;

        fout << "\nFILE " << s.filename << "\n\n";
    }
    fout.close();
}

unsigned ALSMan::NewSeed() {
    boost::uniform_int <> unDistr(numeric_limits <int>::min(), numeric_limits <int>::max());
    unsigned _seed = static_cast <unsigned> (unDistr(randGen));
    cout << "new seed = " << _seed << endl;
    return _seed;
}


bool ALSMan::VerErr(NetMan & net, double err, vector <ll> RealCom) {
    seed = NewSeed();
    double valErr = CalcErr(accNet, net, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
    if (DoubleEqual(err, 0)) {
        double dev = fabs(valErr - err);
        cout << "old " << metrType << " = " << err << ", new " << metrType << " = " << valErr << ", absolute deviation = " << dev << endl;
        if (dev > 0.2) {
            cout << "warning: large deviation of " << metrType << " measurement" << endl;
            return false;
        }
    }
    else {
        // double dev = fabs(err - valErr) / err;
        double dev = (valErr - err) / err;
        cout << "old " << metrType << " = " << err << ", new " << metrType << " = " << valErr << ", relative deviation = " << dev << endl;
        double ref = 0.2;
        if (lacType == LAC_TYPE::CONS)
            ref = 0.2;
        if (dev > ref) {
            cout << "warning: large deviation of " << metrType << " measurement" << endl;
            return false;
        }
    }
    return true;
}

void ALSMan::ApplyLacCon(NetMan & net, std::shared_ptr <LAC> pLac, double backErr) {
    auto p0 = net.CreateOneConst(true);
    auto p1 = net.CreateOneConst(false);
    auto pConstLac = dynamic_pointer_cast <ConstLAC> (pLac);
    auto targId = pConstLac->GetTargId();
    auto err = pConstLac->GetErrPro();
    auto pos = net.GetFanoutsThatArePos(targId);
    cout << net.GetName(targId);
    if (pos.size()) {
        cout << "(driver of ";
        PrintVect(pos);
        cout << ")";
    }
    cout << " is replaced by " << (pConstLac->IsConst0()? "const0": "const1") << " with estimated " << metrType << " " << bigFlt(err) / bigFlt(nFrame) + backErr << endl;
    if (pConstLac->IsConst0())
        net.Replace(targId, p0);
    else
        net.Replace(targId, p1);
}

void ALSMan::ApplyMultLacPro(NetMan & net, std::vector < std::shared_ptr <LAC> > pLacs, double backErr){
    if (lacType == LAC_TYPE::CONS) {
        auto consts = net.CreateConst();
        assert(consts.first != -1 && consts.second != -1);
        ll FirstId = -1;
        auto pCheck0 = dynamic_pointer_cast <ConstLAC> (pLacs[0]);
        auto checkId0 = pCheck0->GetTargId();
        auto pCheck1 = dynamic_pointer_cast <ConstLAC> (pLacs[1]);
        auto checkId1 = pCheck1->GetTargId();
        cout << checkId0 << " and " << checkId1 << endl;
        for (auto i = 0; i < pLacs.size(); ++i){
            auto pConstLac = dynamic_pointer_cast <ConstLAC> (pLacs[i]);
            if (checkId0 > checkId1)
                pConstLac = dynamic_pointer_cast <ConstLAC> (pLacs[1-i]);
            auto targId = pConstLac->GetTargId();
            if (targId == FirstId){
                cout << "Lac1 and Lac2 have the same targId" << endl;
                continue;
            }
            FirstId = targId;
            auto err = pConstLac->GetErrPro();
            auto pos = net.GetFanoutsThatArePos(targId);
            cout << net.GetName(targId);
            if (pos.size()) {
                cout << "(driver of ";
                PrintVect(pos);
                cout << ")";
            }
            consts = net.CreateConst();
            cout << " is replaced by " << (pConstLac->IsConst0()? "const0": "const1") << " with estimated " << metrType << " " << bigFlt(err) / bigFlt(nFrame) + backErr << endl;
            if (pConstLac->IsConst0())
                net.Replace(targId, consts.first);
            else
                net.Replace(targId, consts.second);
        }
    }
    else if (lacType == LAC_TYPE::SASIMI) {
        net.GetLev();
        for (auto & pLac : pLacs){
            auto pSasimiLac = dynamic_pointer_cast <SasimiLAC> (pLac);
            auto targId = pSasimiLac->GetTargId();
            auto subId = pSasimiLac->GetSubId();
            auto isInv = pSasimiLac->GetIsInv();
            auto err = pSasimiLac->GetErrPro();
            cout << "replace " << net.GetObj(targId);
            cout << "(l=" << net.GetObjLev(targId) << ")";
            cout << " by " << net.GetObj(subId);
            cout << "(l=" << net.GetObjLev(subId) << ")";
            cout << "+" << (isInv? "inv": "buf") << " with " << metrType << " " << bigFlt(err) / bigFlt(nFrame) << endl;
            if (!isInv)
                net.Replace(targId, subId);
            else {
                ll newInvId = net.CreateInv(subId);
                net.Replace(targId, newInvId);
            }
        }
    }
}

// wenhui
void ALSMan::ApplyMultLac(NetMan & net, std::vector < std::shared_ptr <LAC> > pLacs, double backErr) {
    // debug
    // net.PrintPro(1, 1, 0);
    // cout << endl;

    if (lacType == LAC_TYPE::CONS) {
        auto consts = net.GetConstId();
        assert(consts.first != -1 && consts.second != -1);
        for (auto i = 0; i < pLacs.size(); ++i) {
            auto pConstLac = dynamic_pointer_cast <ConstLAC> (pLacs[i]);
            auto targId = pConstLac->GetTargId();
            auto err = pConstLac->GetErrPro();
            auto pos = net.GetFanoutsThatArePos(targId);
            cout << net.GetName(targId) << "(oriId = " << net.GetOriId(targId) << ")";
            if (pos.size()) {
                cout << "(driver of ";
                PrintVect(pos);
                cout << ")";
            }
            cout << " is replaced by " << (pConstLac->IsConst0()? "const0": "const1") << " with estimated error increase: " << pConstLac->GetErr() << endl;

            if (pConstLac->IsConst0())
                net.Replace(targId, consts.first);
            else
                net.Replace(targId, consts.second);
        }
    }
    else
        assert(0);
}

double ALSMan::ApplyLacPro(NetMan & net, std::shared_ptr <LAC> pLac, double backErr) {
    double newErr = 0;
    if (lacType == LAC_TYPE::CONS) {
        auto consts = net.GetConstId();
        assert(consts.first != -1 && consts.second != -1);
        auto pConstLac = dynamic_pointer_cast <ConstLAC> (pLac);
        auto targId = pConstLac->GetTargId();
        // auto err = pConstLac->GetErrPro();
        bigFlt err = -1;
        if (metrType == METR_TYPE::MRED) {
            err = pConstLac->GetErrBigFlt();
            // cout << "ErrBigFlt = " << err << endl;
        }
        else
            err = bigFlt(pConstLac->GetErrPro());

        auto pos = net.GetFanoutsThatArePos(targId);
        // cout << net.GetName(targId);
        cout << net.GetName(targId);
        if (pos.size()) {
            cout << "(driver of ";
            PrintVect(pos);
            cout << ")";
        }
        newErr = double(err / bigFlt(nFrame)) + backErr;
        cout << " is replaced by " << (pConstLac->IsConst0()? "const0": "const1") << " with estimated " << metrType << " " << newErr << endl;

        Abc_Obj_t * pNode = net.GetObj(targId);
        cout << "errOrder = " << pNode->order << " with constType = " << pNode->constType << endl;

        if (pConstLac->IsConst0())
            net.Replace(targId, consts.first);
        else
            net.Replace(targId, consts.second);
    }
    else if (lacType == LAC_TYPE::SASIMI) {
        net.GetLev();
        auto pSasimiLac = dynamic_pointer_cast <SasimiLAC> (pLac);
        auto targId = pSasimiLac->GetTargId();
        auto subId = pSasimiLac->GetSubId();
        auto isInv = pSasimiLac->GetIsInv();

        // observe the structure relationship between subNode and targetNode
        net.PrintLocal(1, 1, subId, targId);

        // auto err = pSasimiLac->GetErrPro();
        bigFlt err = -1;
        if (metrType == METR_TYPE::MRED) {
            err = pSasimiLac->GetErrBigFlt();
            // cout << "ErrBigFlt = " << err << endl;
        }
        else
            err = bigFlt(pSasimiLac->GetErrPro());

        // cout << "err = " << err << ", nFrame = " << nFrame << ", backErr = " << backErr << endl;
        newErr = double(err / bigFlt(nFrame)) + backErr;
        
        cout << "targId = " << targId << ", subId = " << subId << endl;
        cout << "replace " << net.GetObj(targId);
        cout << "(l=" << net.GetObjLev(targId) << ")";
        cout << " by " << net.GetObj(subId);
        cout << "(l=" << net.GetObjLev(subId) << ")";
        cout << "+" << (isInv? "inv": "buf") << " with " << metrType << " " << newErr << endl;
        if (!isInv)
            net.Replace(targId, subId);
        else {
            ll newInvId = net.CreateInv(subId);
            net.Replace(targId, newInvId);
        }
    }
    else if (lacType == LAC_TYPE::SUB_WIRE) {
        net.GetLev();
        auto pSubWireLac = dynamic_pointer_cast <SubWireLAC> (pLac);
        auto targId = pSubWireLac->GetTargId();
        auto subId = pSubWireLac->GetSubId();
        auto iFanin = pSubWireLac->GetIFanin();
        auto isInv = pSubWireLac->GetIsInv();
        auto err = pSubWireLac->GetErr();
        cout << "replace " << net.GetObj(targId);
        cout << "(l=" << net.GetObjLev(targId) << ")'s";
        cout << " " << iFanin << "-th fanin, " << net.GetFanin(targId, iFanin);
        cout << "(l=" << net.GetObjLev(net.GetFanin(targId, iFanin)) << "),";
        cout << " by " << net.GetObj(subId);
        cout << "(l=" << net.GetObjLev(subId) << ")";
        assert(metrType != METR_TYPE::SELF);
        cout << "+" << (isInv? "inv": "buf") << " with " << metrType << " " << bigFlt(err) / bigFlt(nFrame) << endl;
        if (!isInv) {
            cout << "original fanins = "; for (int i = 0; i < net.GetFaninNum(targId); ++i) cout << net.GetFanin(targId, i) << ","; cout << endl;
            net.PatchFanin(net.GetObj(targId), iFanin, net.GetFanin(targId, iFanin), net.GetObj(subId));
            cout << "new fanins = "; for (int i = 0; i < net.GetFaninNum(targId); ++i) cout << net.GetFanin(targId, i) << ","; cout << endl;
        }
        else {
            cout << "original fanins = "; for (int i = 0; i < net.GetFaninNum(targId); ++i) cout << net.GetFanin(targId, i) << ","; cout << endl;
            ll newInvId = net.CreateInv(subId);
            net.PatchFanin(net.GetObj(targId), iFanin, net.GetFanin(targId, iFanin), net.GetObj(newInvId));
            cout << "new fanins = "; for (int i = 0; i < net.GetFaninNum(targId); ++i) cout << net.GetFanin(targId, i) << ","; cout << endl;
        }
    }
    else if (lacType == LAC_TYPE::RAC) {
        assert(net.GetNetType() == NET_TYPE::SOP);
        net.GetLev();
        auto pRacLac = dynamic_pointer_cast <RacLAC> (pLac);
        auto targId = pRacLac->GetTargId();
        auto faninIds = pRacLac->GetDivIds();
        auto sop = pRacLac->GetSop();
        auto err = pRacLac->GetErr();
        cout << "replace " << net.GetObj(targId);
        cout << "(l=" << net.GetObjLev(targId) << ") with old fanins (";
        for (ll i = 0; i < net.GetFaninNum(targId); ++i) {
            cout << net.GetFaninId(targId, i) << "(l=" << net.GetObjLev(net.GetFanin(targId, i)) << "),";
        }
        cout << ")";
        cout << " by ";
        cout << "(";
        for (const auto & faninId: faninIds)
            cout << faninId << "(l=" << net.GetObjLev(faninId) << "),";
        cout << ")";
        cout << " with " << metrType << " " << bigFlt(err) / bigFlt(nFrame) << ", function:" << endl;
        cout << sop;
        auto newNodeId = net.CreateNode(faninIds, sop);
        net.Replace(targId, newNodeId);
    }
    else
        assert(0);
    return newErr;
}


void ALSMan::ExactSimpl(NetMan & net, ll round, bool fModifyfGenSub) {
    cout << "***** exactly simplify" << endl;
    cout << "before: ";
    net.PrintStat();

    bool isCont = true;
    while (isCont) {
        net.MergeIdentNode();
        bool isUpd0 = net.CleanUp();
        bool isUpd1 = net.ProcHalfAndFullAdd();
        isCont = (isUpd0 || isUpd1);
    }
    net.CleanUp();
    net.MergeConst();

    // avoid inconsistent node Id
    auto netTmp = net;
    net = netTmp;

    if (net.GetNetType() == NET_TYPE::GATE && propConst) {
        bool isUpd = net.ProcConstInp();
        if (isUpd)
            cout << "Finish gate replacement on const inputs" << endl;
        else
            cout << "No const inputs to be fixed" << endl;
        if (net.GetNetType() == NET_TYPE::GATE && net.CheckSCLNet() == 0)
            net.ReArrInTopoOrd();
        const ll evalRoundInt = 10;
        if (round % evalRoundInt == 0 && net.GetMaxLev() < 500){
            cout << "final:  ";
            net.SynthAndMap(maxDelay, true);
        }
    }
    else {
        if (net.GetNetType() == NET_TYPE::GATE) {
            if (net.CheckSCLNet() == 0)
                net.ReArrInTopoOrd();
            if (net.GetMaxLev() < 500) {
                net.SynthAndMap(maxDelay, true);
            }
        }
        else {
            net.SynthWithResyn2Comm();
            // net.SynthAIG();
        }
    }
}

void ALSMan::Eval(NetMan & net, const std::string & outpPath, double err, ll round) {
    cout << "***** evaluate and output" << endl;
    // if (DoubleGreat(err, errUppBound))
        // return;
    if (net.GetNetType() == NET_TYPE::SOP) {
        // measure and output SOP
        ostringstream oss("");
        oss << outpPath << round << "_" << net.GetNet()->pName << "_" << metrType << "_" << err << "_size_" << net.GetArea() << "_depth_" << net.GetDelay();
        // net.ReArrInTopoOrd();
        // net.WriteNet(oss.str() + ".blif");
        net.WriteNet(oss.str() + ".v", true);
        // net.DumpCFile(oss.str() + ".c");

        // measure and output the mapped network
        auto mapNet = net;
        cout << "run st; compress2rs; ps; dch; amap;" << endl;
        mapNet.Comm("st; compress2rs; ps; dch; amap;", true);
        mapNet.ReArrInTopoOrd();
        ostringstream oss2("");
        oss2 << outpPath << round << "_" << mapNet.GetNet()->pName << "_" << metrType << "_" << err << "_area_" << mapNet.GetArea() << "_delay_" << mapNet.GetDelay();
        mapNet.WriteNet(oss2.str() + ".v", true);
    }
    else if (net.GetNetType() == NET_TYPE::GATE) {
        // measure and output original gate-netlist
        auto tempNet = net;
        tempNet.ReArrInTopoOrd();
        ostringstream oss("");
        oss << outpPath << round << "_" << tempNet.GetNet()->pName << "_" << metrType << "_" << err << "_area_" << tempNet.GetArea() << "_delay_" << tempNet.GetDelay();
        tempNet.WriteNet(oss.str() + ".v", true);
        // tempNet.DumpCFile(oss.str() + ".c");
        // tempNet.ConvToSop();
        // tempNet.WriteNet(oss.str() + ".blif", true);
    }
    else
        assert(0);
}

void ALSMan::Eval_app(NetMan & net, const std::string & outpPath, double err, ll round) {
    cout << "***** evaluate and output" << endl;
    // if (DoubleGreat(err, errUppBound))
        // return;
    if (net.GetNetType() == NET_TYPE::SOP) {
        // measure and output SOP
        ostringstream oss("");
        oss << outpPath << round << "_" << appCktName << "_" << metrType << "_" << err << "_size_" << net.GetArea() << "_depth_" << net.GetDelay();
        // net.ReArrInTopoOrd();
        net.WriteNet(oss.str() + ".blif");
        net.WriteNet(oss.str() + ".v");
        net.DumpCFile(oss.str() + ".c");
    }
    else if (net.GetNetType() == NET_TYPE::GATE) {
        // measure and output original gate-netlist
        auto tempNet = net;
        tempNet.ReArrInTopoOrd();
        ostringstream oss("");
        oss << outpPath << round << "_" << appCktName << "_" << metrType << "_" << err << "_area_" << tempNet.GetArea() << "_delay_" << tempNet.GetDelay();
        tempNet.WriteNet(oss.str() + ".v", true);
        // tempNet.DumpCFile(oss.str() + ".c");
        // tempNet.ConvToSop();
        // tempNet.WriteNet(oss.str() + ".blif", true);
    }
    else
        assert(0);
}

void ALSMan::EvalPro(NetMan & net, const std::string & outpPath, double err, ll round, const std::string mark) {
    cout << "***** evaluate and output" << endl;
    // if (DoubleGreat(err, errUppBound))
        // return;
    if (net.GetNetType() == NET_TYPE::SOP) {
        // measure and output SOP
        ostringstream oss("");
        oss << outpPath << round << "_" << net.GetNet()->pName << "_" << metrType << "_" << err << "_size_" << net.GetArea() << "_depth_" << net.GetDelay() << "_" << mark;
        // net.ReArrInTopoOrd();
        net.WriteNet(oss.str() + ".blif");
        net.WriteNet(oss.str() + ".v");
        net.DumpCFile(oss.str() + ".c");
    }
    else if (net.GetNetType() == NET_TYPE::GATE) {
        // measure and output original gate-netlist
        auto tempNet = net;
        tempNet.ReArrInTopoOrd();
        ostringstream oss("");
        oss << outpPath << round << "_" << tempNet.GetNet()->pName << "_" << metrType << "_" << err << "_area_" << tempNet.GetArea() << "_delay_" << tempNet.GetDelay() << "_" << mark;
        tempNet.WriteNet(oss.str() + ".v", true);
        // tempNet.DumpCFile(oss.str() + ".c");
        // tempNet.ConvToSop();
        // tempNet.WriteNet(oss.str() + ".blif", true);
    }
    else
        assert(0);
}


vector < vector <ll> > ALSMan::TempApplyLacs(NetMan & net, vector < std::shared_ptr <LAC> > & lacs, LAC_TYPE lacType, bool isVerb) {
    #ifdef DEBUG
    assert(lacs.size());
    #endif
    vector < vector <ll> > replTraces;
    for (const auto & pLac: lacs) {
        Abc_Obj_t * pTS = nullptr, * pSS = nullptr;
        pTS = net.GetObj(pLac->GetTargId());

        if (lacType == LAC_TYPE::CONS) {
            auto constLac = *dynamic_pointer_cast <ConstLAC> (pLac);
            auto consts = net.GetConstId();
            pSS = constLac.IsConst0()? net.GetObj(consts.first): net.GetObj(consts.second);
            if (isVerb) cout << pTS << " is replaced by const" << !constLac.IsConst0() << endl;
        }
        else if (lacType == LAC_TYPE::SASIMI) {
            auto sasimiLac = *dynamic_pointer_cast <SasimiLAC> (pLac);
            auto pSub = net.GetObj(sasimiLac.GetSubId());
            pSS = sasimiLac.GetIsInv()? net.CreateInv(pSub): pSub;
            if (isVerb) cout << pTS << " is replaced by " << (sasimiLac.GetIsInv()? "\\bar ": "") << pSS << endl;
        }
        else if (lacType == LAC_TYPE::RAC) {
            auto racLac = *dynamic_pointer_cast <RacLAC> (pLac);
            auto newNodeId = net.CreateNode(racLac.GetDivIds(), racLac.GetSop());
            pSS = net.GetObj(newNodeId);
            if (isVerb) {cout << pTS << " is rewritten, "; net.PrintObj(pSS, true);}
        }
        else
            assert(0);
        auto replTrace = net.TempRepl(pTS, pSS);
        replTraces.emplace_back(replTrace);
    }
    return replTraces;
}

bool ALSMan::RunALS(const std::string & outpPath) {
    ExactSimpl(accNet, 0, 0);   // for non-gateNetlist: Comm("st; logic; sop; ps;");
    auto currNet = accNet;
    if (pAppNtk != nullptr) {
        NetMan iniAppNet(pAppNtk, true);
        currNet = iniAppNet;
    }

    // set compensation to 0
    vector <ll> RealCom;
    for (auto i = 0; i < nOutput; ++i)
        RealCom.emplace_back(0);
    assert(currNet.GetPoNum() % nOutput == 0);

    // print basic initial information
    ll round = 1;
    cout << endl << "network representation: " << accNet.GetNetType() << endl;  
    cout << "PI num = " << accNet.GetPiNum() << ", PO num = " << accNet.GetPoNum() << endl;
    cout << "node num = " << accNet.GetNodeNum() << endl;
    cout << "level = " << accNet.GetLev() << endl;
    // check if all sop nodes have exactly 2 inputs (actually AIG format)
    if (accNet.GetNetType() == NET_TYPE::SOP) {
        bool flag = true;
        for (auto nodeId : accNet.TopoSortWithIds()) {
            if (!accNet.IsNode(nodeId))
                continue; 
            if (accNet.IsConst(nodeId))
                continue;      
            if (accNet.GetFaninNum(nodeId) != 2) {
                flag = false;
                cout << "node " << nodeId << " has " << accNet.GetFaninNum(nodeId) << " inputs" << endl;
                break;
            }
        }
        if (flag) {
            cout << "all sop nodes have exactly 2 inputs (actually AIG format)" << endl;
        }
        else {
            cout << "not all sop nodes have exactly 2 inputs (not AIG format)" << endl;
        }
    }
    // calculate initial error
    double err = CalcErr(accNet, currNet, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
    cout << "initial " << metrType << " = " << err << endl;
    if (maxDelay != numeric_limits <double>::max())
        cout << "max delay = " << maxDelay << endl;
    maxArea = accNet.GetArea();
    // output accNet   
    Eval(accNet, outpPath, 0.0, 0);

    // main loop
    auto start = chrono::system_clock::now();

    cout << "*************************** applying approx rewriting ***************************" << endl;
    ll roundRW = round;
    while (DoubleLessEqual(err, errUppBound) || round == roundRW) {
        auto roundStart = chrono::system_clock::now();
        cout << "----------------- round " << round << "----------------- " << endl;
        err = SimplByWinRewrite(currNet, outpPath, round, RealCom);
        auto duration = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - roundStart);
        cout << "runtime for this round = " << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " sec" << endl;
        duration = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - start);
        cout << "total actual runtime = " << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " sec" << endl;
        ++round;
        // db.UpdateSuppDB();
    }
    db.UpdateSuppDB();
    return 0;
}

extern "C" {
    void Abc_SclTimePerformIntPro(SC_Lib * pLib, Abc_Ntk_t * pNtk, int nTreeCRatio, int fUseWireLoads, int fShowAll, int fPrintPath, int fDumpStats, int fPrint);
}
double ALSMan::SimplByWinRewrite(NetMan & net, const std::string & outpPath, ll round, std::vector <ll> RealCom) {
    // backup network
    // auto backNet = net;

    // calculate level
    ll lev = net.GetLev();
    cout << "lev of currNet = " << lev << endl;
    net.SetLevel(lev);

    // use new seed
    // if (seed == 0)
    seed = NewSeed();   
    cout << "use seed " << seed << endl;
    // auto backErr = CalcErr(accNet, net, isSign, seed, 4*nFrame, nOutput, metrType, distrType, RealCom, -1);
    auto backErr = CalcErr(accNet, net, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
    cout << "backup " << metrType << " = " << backErr << endl;
    double backArea = net.GetArea();
    double backDelay = net.GetDelay();
    maxDelay = backDelay;
    
    // create constant nodes
    net.CreateConst(true);
    // auto tmpNet = net;      // avoid inconsistent const nodes' Id
    // net = tmpNet;

    // get target nodes
    auto nodes = net.TopoSortWithIds();
    cout << "#nodes = " << nodes.size() << endl;
    // net.PrintPro(1, 1, 0);
    // cout << endl;

    // generate LACs
    LACMan lacMan;
    // lacType = LAC_TYPE::CONS;
    // lacMan.GenConstLACs(net, nodes);
    lacType = LAC_TYPE::SASIMI;
    lacMan.GenSasimiLACsNew(net, nodes);
    cout << "#lacs = " << lacMan.GetLacNum() << endl;

    // error estimation
    #ifdef DEBUG
    assert(IsPIOSame(accNet, net));
    #endif
    Simulator accSmlt(accNet, seed, nFrame);
    Simulator appSmlt(net, seed, nFrame);
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

    // const bigInt uppBound = bigInt(nFrame) * bigInt(errUppBound);
    const bigInt uppBound = bigInt(nFrame * errUppBound * 2);
    VECBEEMan vecbeeMan(isSign, seed, nFrame, nOutput, metrType, lacType, distrType, nThread);
    // vecbeeMan.BatchErrEst(net, accSmlt, appSmlt, lacMan, uppBound, nOutput, RealCom, nCand);
    auto startBatchErr = chrono::system_clock::now();
    vecbeeMan.BatchErrEst(net, accSmlt, appSmlt, lacMan, uppBound, nOutput, RealCom, lacMan.GetLacNum());
    auto duration = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - startBatchErr);
    cout << "runtime for BatchErrEst = " << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " sec" << endl;

    cout << "consider single sasimi LAC: " << endl;
    double newErr2 = errUppBound * 2;
    auto backNet2 = net;
    auto net2 = backNet2;
    lacMan.SortLacs(metrType);
    cout << "#sasimi LAC = " << lacMan.GetLacNum() << endl;
    std::shared_ptr<LAC> pBestLac;
    std::shared_ptr<LAC> pBestLac0;
    double newErr0;
    bool fFail = true;
    for (ll i = 0; i < lacMan.GetLacNum(); ++i) {
        cout << "i = " << i << ": ";
        pBestLac = lacMan.GetLac(i);
        
        auto currNet = net2;
        newErr2 = ApplyLacPro(net2, pBestLac, backErr);
        // avoid inconsistent node Id
        auto netTmp = net2;
        net2 = netTmp;

        if (i == 0) {
            pBestLac0 = pBestLac;
            newErr0 = newErr2;
        }
        net2.ReArrInTopoOrd();
        if (net2.CheckSCLNet() == 0) {
            cout << "fail to sort in topo order!" << endl;

            cout << "currNet(before applying): " << endl;
            currNet.PrintPro(1, 1, 0);

            cout << "net2(after applying): " << endl;
            net2.PrintPro(1, 1, 0);
            
            cout << "CheckSasimiLev(backNet):" << endl;
            assert(lacMan.CheckSasimiLev(backNet2));
            cout << endl << "CheckSasimiLev(currNet):" << endl;
            assert(lacMan.CheckSasimiLev(currNet));
            assert(0);
        }
        if (newErr2 > errUppBound) {
            cout << "exceed errUppBound!" << endl;
            break;
        }
        double delay = net2.GetDelay();
        cout << "new area = " << net2.GetArea() << ", new delay = " << delay << endl;
        if (delay <= maxDelay) {
            cout << "use seed " << seed << endl;
            newErr2 = CalcErr(accNet, net2, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
            cout << "apply SASIMI LAC: error(simulation) = " << newErr2 << endl;
            fFail = false;
            break;
        }
        else {
            net2 = backNet2;
        }
    }
    if (fFail && (newErr2 > errUppBound) && (newErr0 <= errUppBound)) {
        net2 = backNet2;
        ApplyLacPro(net2, pBestLac0, backErr);
        // avoid inconsistent node Id
        auto netTmp = net2;
        net2 = netTmp;
        net2.ReArrInTopoOrd();

        cout << "original(backNet) area = " << backNet2.GetArea() << ", new area = " << net2.GetArea() << endl;
        cout << "use seed " << seed << endl;
        newErr2 = CalcErr(accNet, net2, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
        cout << "apply SASIMI LAC: error(simulation) = " << newErr2 << endl;
    }
    double newArea2 = net2.GetArea();
    double newDelay2 = net2.GetDelay();
    if (newErr2 == 0 && newArea2 < maxArea && newDelay2 <= maxDelay) {
        cout << "Due to zero error, directly apply sasimi LAC in this round" << endl;
        cout << "newArea2 = " << newArea2 << ", newDelay2 = " << newDelay2 << endl;
        net = net2;
        ExactSimpl(net, round, 0);
        // avoid inconsistent node Id
        auto netTmp = net;
        net = netTmp;
        Eval(net, outpPath, newErr2, round);
        return newErr2;
    }

    // select Scand
    set<ll> candSet = lacMan.GetScand(nCand, metrType, net, nFrame);
    vector <ll> Scand(candSet.begin(), candSet.end());
    cout << "Scand size = " << Scand.size() << endl;

    // select LO and generate sub-circuits
    auto startGenSubCkts = chrono::system_clock::now();
    ll maxAppRWNum = 500;   // can be tuned
    SubCktMan subcktMan(net, metrType, nOutput, nFrame, isSign, errUppBound, backErr, maxAppRWNum);
    subcktMan.GenSubCkts(Scand);
    duration = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - startGenSubCkts);
    cout << "runtime for GenSubCkts = " << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " sec" << endl;
    cout << "#subCkts(#LO = 2) = " << subcktMan.GetSubCktNum2();
    cout << ", #subCkts(#LO = 3) = " << subcktMan.GetSubCktNum3() << endl;
    cout << "#trivial subCkts(#LO = 2) = " << subcktMan.GetSubCktNum2T();
    cout << ", #trivial subCkts(#LO = 3) = " << subcktMan.GetSubCktNum3T() << endl;

    // calculate multi-LO's boolean difference
    auto startCalcBD = chrono::system_clock::now();
    subcktMan.CalcBD(appSmlt, vecbeeMan.getBdPo2Nodes(), vecbeeMan.getPoMarks(), vecbeeMan.getTopoIds());
    cout << "finish CalcBD!" << endl;
    duration = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - startCalcBD);
    cout << "runtime for calculate multi-LO's bd = " << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " sec" << endl;

    // generate divisor sets
    subcktMan.GenDivs();
    cout << "finish GenDivs!" << endl;

    // generate approximate catalog for each sub-circuit
    // auto startGenAppRW = chrono::system_clock::now();
    subcktMan.GenAllAppRWsPro(appSmlt, accSmlt, db, vecbeeMan.getBdPo2Nodes(), nThread);
    // duration = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - startGenAppRW);
    // cout << "runtime for GenAppRWs = " << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " sec" << endl;
    cout << "#AppRW = " << subcktMan.GetAppRwNum() << endl;

    // evaluate error and filter
    subcktMan.BatchErr(appSmlt, accSmlt, vecbeeMan.getBdPo2Nodes(), nThread);
    cout << "#AppRW = " << subcktMan.GetAppRwNum() << endl;

    auto startSelectBestRW = chrono::system_clock::now();
    std::shared_ptr <AppRW> pBestRW = SelectBestRW(net, subcktMan, backArea, backErr);
    duration = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now() - startSelectBestRW);
    cout << "runtime for SelectBestRW = " << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << " sec" << endl;

    double newErr = errUppBound * 2;
    bool fUseRW = true;    // 1: use RW (default); 0: use sasimi
    if (pBestRW == nullptr) {
        cout << "pBestRW is nullptr" << endl;
        fUseRW = false;
    }
    else {
        // apply best RW to net to obtain performance metrics
        pBestRW->Print();
        pBestRW->PrintOriSubNtk(net);   // print previous sub-circuit
        ApplyRW(net, pBestRW, backArea, true);
        auto netTmp = net;  // avoid inconsistent node Id
        net = netTmp;
        net.ReArrInTopoOrd();
        // check error by simulation
        newErr = CalcErr(accNet, net, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
        cout << "apply appRW: error(simulation) = " << newErr << ", error(VECBEE) = " << pBestRW->GetError() << endl;
        double newArea = net.GetArea();
        double newDelay = net.GetDelay();
        
        // compare appRW & sasimi
        cout << "backErr = " << backErr << ", backArea = " << backArea << ", backDelay = " << backDelay << ", maxDelay = " << maxDelay << endl;
        cout << "newErr(appRW) = " << newErr << ", newErr2(sasimi) = " << newErr2 << endl;
        cout << "newArea(appRW) = " << newArea << ", newArea(sasimi) = " << newArea2 << endl;
        cout << "newDelay(appRW) = " << newDelay << ", newDelay(sasimi) = " << newDelay2 << endl;

        // check sasimi's score
        if (newArea2 < backArea && newDelay2 <= maxDelay) {
            // double scoreRW = (newErr - backErr)/((backArea - newArea)/maxArea + (maxDelay - newDelay)/maxDelayOri);
            // double scoreSASIMI = (newErr2 - backErr)/((backArea - newArea2)/maxArea + (maxDelay - newDelay2)/maxDelayOri);
            double scoreRW = CalcScore(backArea, newArea, maxDelay, newDelay, maxArea, maxDelayOri, newErr - backErr);
            double scoreSASIMI = CalcScore(backArea, newArea2, maxDelay, newDelay2, maxArea, maxDelayOri, newErr2 - backErr);
            cout << "score(appRW) = " << scoreRW << ", score(sasimi) = " << scoreSASIMI << endl;
            if (scoreSASIMI == scoreRW) {
                if (newArea2 < newArea)
                    fUseRW = false;
            }
            else if (scoreSASIMI < scoreRW)
                fUseRW = false;
        }
    }

    if (fUseRW)
        cout << "use appRW in this round" << endl;
    else {
        cout << "use sasimi in this round" << endl;
        net = net2;
        newErr = newErr2;
    }
   
    // measure, synthesis & mapping, output
    if (DoubleLessEqual(newErr, errUppBound)) {
        ExactSimpl(net, round, 0);
        // avoid inconsistent node Id
        auto netTmp = net;
        net = netTmp;

        Eval(net, outpPath, newErr, round);
    }

    return newErr;
}

bool ALSMan::ApplyRW(NetMan & net, std::shared_ptr <AppRW> pBestRW, double backArea, bool fPrint) {
    ll nVars = pBestRW->GetnVars();
    vector <string> sopFuncs;
    vector <ll> LoIndex(pBestRW->GetnLOs(), -1);    // LoIndex[i] is the corresponding PO id in pSynNtk for the i-th LO

    // build truth table
    ll o = 0;
    for (const auto & tableValue: pBestRW->GetFuncs()) {
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


        auto it = find(sopFuncs.begin(), sopFuncs.end(), func);
        if (it != sopFuncs.end()) {     // duplication exists
            LoIndex[o] = static_cast <ll> (it - sopFuncs.begin());
        } 
        else {
            sopFuncs.push_back(func);
            LoIndex[o] = sopFuncs.size() - 1;   // the last index
        }
        ++o;
    }

    // Synthesis sop func as a new network
    Abc_Ntk_t * pSubNtk = Abc_NtkAlloc(ABC_NTK_LOGIC, ABC_FUNC_SOP, 1);
    Abc_Obj_t ** pFaninNodes = new Abc_Obj_t*[nVars];
    for (int k = 0; k < nVars; ++k) {
        pFaninNodes[k] = Abc_NtkCreatePi(pSubNtk);
    }
    for (string func : sopFuncs) {
        Abc_Obj_t * pNewNode = Abc_NtkCreateNode(pSubNtk);
        if (func != "")
            pNewNode->pData = Abc_SopRegister((Mem_Flex_t *)pSubNtk->pManFunc, func.c_str());
        else 
            pNewNode->pData = Abc_SopCreateConst0((Mem_Flex_t *)pSubNtk->pManFunc);
        for (int k = 0; k < nVars; ++k) {
            Abc_ObjAddFanin(pNewNode, pFaninNodes[k]);
        }
        Abc_Obj_t * pOutNode = Abc_NtkCreatePo(pSubNtk);
        Abc_ObjAddFanin(pOutNode, pNewNode);    
    }
    // synthesis
    Abc_Frame_t * pAbc = Abc_FrameGetGlobalFrame();
    Abc_FrameReplaceCurrentNetwork(pAbc, Abc_NtkDup(pSubNtk));
    string Command = string("strash; resyn2a; logic; amap; topo;");
    assert(!Cmd_CommandExecute(pAbc, Command.c_str()));
    Abc_NtkDelete(pSubNtk);
    Abc_Ntk_t * pSynNtk = Abc_NtkDup(Abc_FrameReadNtk(pAbc));

    if (fPrint) {
        cout << "new sub-circuit:" << endl;
        PrintNtk(pSynNtk);
    }

    // double area = Abc_NtkGetMappedArea(pSynNtk);
    // if (fPrint)
    //     cout << "real new(app) sub-circuit area = " << area << endl;
    bool fArea = false;
    // if (!DoubleEqual(area, pBestRW->GetOriArea() - pBestRW->GetReArea())) {
    //     if (fPrint)
    //         cout << "real new(app) sub-circuit area (= " << area << ") are different from estimated area (= " << (pBestRW->GetOriArea() - pBestRW->GetReArea()) << "), oriArea = " << pBestRW->GetOriArea() << endl;

    //     if (DoubleGreatEqual(area, pBestRW->GetOriArea()))
    //         fArea = true;
    // }
    // if ((pBestRW->GetReArea() == 0) && (area != 0)) {
    //     if (fPrint) 
    //         cout << "the estimated area is unknown" << endl;
    //     if (DoubleGreatEqual(area, pBestRW->GetOriArea()))
    //         fArea = true;
    // }

    // insert the sub-circuit into appNet, and delete original sub-circuit
    net.ReplaceSubCkt(pBestRW->GetvDiv(), pBestRW->GetvLO(), pSynNtk, LoIndex);

    double newArea = net.GetArea();
    // if (DoubleGreatEqual(newArea, backArea))    
    if (backArea - newArea < 0.1)   // can be tuned
        fArea = true;

    // clean up
    delete[] pFaninNodes;
    Abc_NtkDelete(pSynNtk);

    return fArea;
}

void ALSMan::ObserveArea() {
    db.loadNonCanoDB();

    db.ObserveNonCanoFuncArea(1799);
    cout << endl;

    db.ObserveNonCanoFuncArea(4383);
    cout << endl;

    db.ObserveNonCanoFuncArea(4369);
    cout << endl;

    db.ObserveNonCanoFuncArea(21845);
    cout << endl;

    db.ObserveNonCanoFuncArea(393);
    cout << endl;
}

void ALSMan::GenDB_appFunc() {
    db.GenAppFuncDB(4);
}

std::shared_ptr <AppRW> ALSMan::SelectBestRW(NetMan & net, SubCktMan & subcktMan, double backArea, double backErr) {
    subcktMan.SortAppRWsByErr();
    ll AppRwNum = subcktMan.GetAppRwNum();
    cout << "after SortAppRWsByErr, #AppRW = " << AppRwNum << endl;
    if (AppRwNum == 0)
        return nullptr;

    vector <std::shared_ptr <AppRW>> pFeasAppRWs;   // feasible: area benifit > 0 && delay <= maxDelay
    ll feasNum = 100;   // can be tuned

    for (ll i = 0; i < AppRwNum; ++i) {          
        auto pRW = subcktMan.GetAppRW(i);
        // apply and check area
        auto tmpNet = net;
        bool fAreaErr = ApplyRW(tmpNet, pRW, backArea);
        if (fAreaErr)
            continue;
        // avoid inconsistent node Id
        auto tmpNet2 = tmpNet;
        tmpNet = tmpNet2;
        // check delay
        tmpNet.ReArrInTopoOrd();
        if (tmpNet.CheckSCLNet() == 0) {
            cout << "fail to sort in topo order!" << endl;
            cout << "net: " << endl;
            net.PrintPro(1, 1, 0);
            cout << "tmpNet: " << endl;
            tmpNet.PrintPro(1, 1, 0);          
            assert(0);
        }
        double newDelay = tmpNet.GetDelay();
        if (newDelay > maxDelay) 
            continue;
        
        // calculate score
        double newArea = tmpNet.GetArea();
        assert(newArea < backArea);
        // double score = (pRW->GetError() - backErr)/((backArea - newArea)/maxArea + (maxDelay - newDelay)/maxDelayOri);    
        double score = CalcScore(backArea, newArea, maxDelay, newDelay, maxArea, maxDelayOri, pRW->GetError() - backErr);
        pRW->SetScore(score);
        pRW->SetReArea(backArea - newArea);     // update accurate reArea
        pRW->SetReDelay(maxDelay - newDelay);
        pFeasAppRWs.push_back(pRW);
        if (pFeasAppRWs.size() >= feasNum)
            break;
    }
    cout << "pFeasAppRWs.size() = " << pFeasAppRWs.size() << endl;
    if (pFeasAppRWs.empty())
        return nullptr;

    // sort by score (smaller is better)
    sort(pFeasAppRWs.begin(), pFeasAppRWs.end(), [](const std::shared_ptr<AppRW>& a, const std::shared_ptr<AppRW>& b) {
        if (a->GetScore() == b->GetScore()) {
            if (a->GetReArea() == b->GetReArea())
                return a->GetReDelay() > b->GetReDelay();
            return a->GetReArea() > b->GetReArea();
        }
        return a->GetScore() < b->GetScore();
    });

    return pFeasAppRWs[0];
}

double CalcScore(double oldArea, double newArea, double oldDelay, double newDelay, double accArea, double accDelay, double deltaErr) {
    double weightA = 0.5;   // can be tuned
    double weightD = 1 - weightA;

    double benifitPercA = (oldArea - newArea)/accArea;
    double benifitPercD = (oldDelay - newDelay)/accDelay;
    if (deltaErr > 0)
        return deltaErr/(weightA * benifitPercA + weightD * benifitPercD);
    else if (deltaErr < 0)
        return deltaErr * (weightA * benifitPercA + weightD * benifitPercD);
    else
        return 0.0;
}   

void ALSMan::MeasureCkt(const std::string & outpPath) {
    // ExactSimpl(accNet, 0, 0);
    cout << "accNet: area = " << accNet.GetArea() << ", delay = " << accNet.GetDelay() << endl;
    assert(pAppNtk != nullptr);
    NetMan appNet(pAppNtk, true);

    // set compensation to 0
    // vector <ll> RealCom;
    // for (auto i = 0; i < nOutput; ++i)
    //     RealCom.emplace_back(0);
    // assert(appNet.GetPoNum() % nOutput == 0);

    // double err = CalcErr(accNet, appNet, isSign, seed, nFrame, nOutput, metrType, distrType, RealCom, -1);
    // cout << metrType << " = " << err << endl;
    // // output appNet   
    // Eval_app(appNet, outpPath, err, 0);

    appNet.SynthAndMap_v2(maxDelay, true);
}