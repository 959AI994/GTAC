#include "cmdline.hpp"
#include "header.h"
#include "my_abc.h"
#include "als.h"


using namespace abc;
using namespace boost;
using namespace cmdline;
using namespace std;


parser CommPars(int argc, char * argv[]) {
    parser option;
    option.add <ll> ("mode", '\0', "1: graph partition", true, 1);
    option.add <string> ("accCirc", '\0', "path to accurate circuit", true);
    option.add <string> ("appCirc", '\0', "path to approximate circuit", false, "");
    option.add <string> ("standCell", '\0', "path to standard cell library", false, "library/ASAP7_RVT_TT_nldm_NO_FAHA.lib");
    // option.add <string> ("standCell", '\0', "path to standard cell library", false, "library/asap7.lib");
    option.add <string> ("outpPath", '\0', "path to approximate circuits", false, "tmp");
    option.add <string> ("metrType", '\0', "error metric type: ER, MED, NMED, MSE, SNR, SELF", false, "MSE");
    option.add <string> ("distrType", '\0', "error distribution type: UNIF, ENUM, MIX(only for signed 9x8 multiplier), SELF", false, "UNIF");
    option.add <unsigned> ("seed", '\0', "seed for randomness", false, 0);
    option.add <double> ("errUppBound", '\0', "error upper bound", false, 0.1);
    option.add <ll> ("nFrame", '\0', "#Monte Carlo samples, nFrame should be an integer multiple of 64", false, 100032);
    option.add <ll> ("nThread", '\0', "number of threads", false, 1);
    option.add <ll> ("nOutput", '\0', "the number we split the output", false, 1, cmdline::range(1, 16));
    option.add("isSign", '\0', "whether the circuit outputs a signed number or not");
    option.add("inpMapVerilog", '\0', "whether the input file is a gate netlist in Verilog or not");
    option.parse_check(argc, argv);
    return option;
}


void ALS(ALSOpt & alsOpt) {
    alsOpt.Print();
    ALSMan alsMan(alsOpt);

    if (alsOpt.workMode == 1)
        alsMan.GraphPartition();
    else if (alsOpt.workMode == 2) {
        // alsMan.GraphMerge();
        // alsMan.GraphMerge_greedy();
        alsMan.GraphMerge_binary();
    }
    else
        assert(0);
}


int main(int argc, char * argv[]) {
    GlobStartAbc();

    parser option = CommPars(argc, argv);
    ALSOpt alsOpt;
    alsOpt.workMode = option.get <ll> ("mode");
    string accCirc = option.get <string> ("accCirc");
    string appCirc = option.get <string> ("appCirc");
    string standCell = option.get <string> ("standCell");
    alsOpt.outpPath = option.get <string> ("outpPath");
    string metrType = option.get <string> ("metrType");
    string distrType = option.get <string> ("distrType");
    alsOpt.sourceSeed = option.get <unsigned> ("seed");
    alsOpt.errUppBound = option.get <double> ("errUppBound");
    alsOpt.nFrame = option.get <ll> ("nFrame");
    alsOpt.nThread = option.get <ll> ("nThread");
    alsOpt.nOutput = option.get <ll> ("nOutput");
    alsOpt.isSign = option.exist("isSign");
    alsOpt.inpMapVerilog = option.exist("inpMapVerilog");
    
    AbcMan abcMan;
    if (standCell != "")
        abcMan.ReadStandCell(standCell);
    abcMan.ReadNet(accCirc, alsOpt.inpMapVerilog);
    alsOpt.pNtk = Abc_NtkDup(abcMan.GetNet());
    std::filesystem::path pAcc(accCirc);
    alsOpt.accCktName = pAcc.stem().string();  // stem() drops extension; filename() is the last path component
    cout << "alsOpt.accCktName = " << alsOpt.accCktName << endl;
    // cout << "Abc_NtkHasMapping(alsOpt.pNtk) = " << Abc_NtkHasMapping(alsOpt.pNtk) << endl; 

    if (appCirc != "") {
        abcMan.ReadNet(appCirc, alsOpt.inpMapVerilog);
        alsOpt.pAppNtk = Abc_NtkDup(abcMan.GetNet());

        std::filesystem::path p(appCirc);
        alsOpt.appCktName = p.stem().string();  // stem() drops extension; filename() is the last path component
    }
    else {
        alsOpt.pAppNtk = nullptr;
        alsOpt.appCktName = "";
    }
    // cout << "Abc_NtkHasMapping(alsOpt.pNtk) = " << Abc_NtkHasMapping(alsOpt.pNtk) << endl; 

    FixPath(alsOpt.outpPath);
    CreatePath(alsOpt.outpPath);
    if (alsOpt.sourceSeed == 0) {
        random::mt19937 rng(time(0));
        boost::uniform_int <> unif(INT_MIN, INT_MAX);
        alsOpt.sourceSeed = static_cast <unsigned> (unif(rng));
    }
    if (distrType == "UNIF")
        alsOpt.distrType = DISTR_TYPE::UNIF;
    else if (distrType == "ENUM") {
        alsOpt.distrType = DISTR_TYPE::ENUM;
        #ifdef DEBUG
        assert(Abc_NtkPiNum(abcMan.GetNet()) < 20);
        #endif
        alsOpt.nFrame = 1ll << Abc_NtkPiNum(abcMan.GetNet());
        cout << "nFrame for enumeration = " << alsOpt.nFrame << endl;
    }
    else if (distrType == "MIX")
        alsOpt.distrType = DISTR_TYPE::MIX;
    else
        assert(0);
    
    if (metrType == "ER")
        alsOpt.metrType = METR_TYPE::ER;
    else if (metrType == "MED")
        alsOpt.metrType = METR_TYPE::MED;
    else if (metrType == "NMED") {
        alsOpt.metrType = METR_TYPE::MED;
        auto nPo = Abc_NtkPoNum(abcMan.GetNet());
        #ifdef DEBUG
        // assert(nPo <= 60);
        assert(nPo <= 130);
        #endif
        // alsOpt.errUppBound *= ((1ll << nPo) - 1);
        alsOpt.errUppBound *= double((bigInt(1) << nPo) - 1);
        cout << "nPo = " << nPo << ", MED uppBound = " << alsOpt.errUppBound << endl;
    }
    else if (metrType == "MSE")
        alsOpt.metrType = METR_TYPE::MSE;
    else if (metrType == "SNR") {
        alsOpt.metrType = METR_TYPE::MSE;
        NetMan tempNet(abcMan.GetNet(), true);
        alsOpt.errUppBound = GetMSEFromSNR(tempNet, alsOpt.isSign, alsOpt.sourceSeed, alsOpt.nFrame, alsOpt.distrType, alsOpt.errUppBound, alsOpt.nOutput);
    }
    else if (metrType == "MRED") {
        alsOpt.metrType = METR_TYPE::MRED;
    }
    else if (metrType == "MHD") {
        alsOpt.metrType = METR_TYPE::MHD;
    }
    else if (metrType == "NMHD") {
        alsOpt.metrType = METR_TYPE::MHD;
        auto nPo = Abc_NtkPoNum(abcMan.GetNet());
        // #ifdef DEBUG
        // // assert(nPo <= 60);
        // assert(nPo <= 130);
        // #endif
        alsOpt.errUppBound *= double(nPo);
        cout << "nPo = " << nPo << ", MHD uppBound = " << alsOpt.errUppBound << endl;
    }
    else
        assert(0);

    ALS(alsOpt);

    GlobStopAbc();
    return 0;
}