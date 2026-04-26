// Benchmark "../EPFL/benchmarks/random_control/router" written by ABC on Mon Nov  3 16:03:15 2025

module router  ( 
    \dest_x[0] , \dest_x[1] , \dest_x[2] , \dest_x[3] , \dest_x[4] ,
    \dest_x[5] , \dest_x[6] , \dest_x[7] , \dest_x[8] , \dest_x[9] ,
    \dest_x[10] , \dest_x[11] , \dest_x[12] , \dest_x[13] , \dest_x[14] ,
    \dest_x[15] , \dest_x[16] , \dest_x[17] , \dest_x[18] , \dest_x[19] ,
    \dest_x[20] , \dest_x[21] , \dest_x[22] , \dest_x[23] , \dest_x[24] ,
    \dest_x[25] , \dest_x[26] , \dest_x[27] , \dest_x[28] , \dest_x[29] ,
    \dest_y[0] , \dest_y[1] , \dest_y[2] , \dest_y[3] , \dest_y[4] ,
    \dest_y[5] , \dest_y[6] , \dest_y[7] , \dest_y[8] , \dest_y[9] ,
    \dest_y[10] , \dest_y[11] , \dest_y[12] , \dest_y[13] , \dest_y[14] ,
    \dest_y[15] , \dest_y[16] , \dest_y[17] , \dest_y[18] , \dest_y[19] ,
    \dest_y[20] , \dest_y[21] , \dest_y[22] , \dest_y[23] , \dest_y[24] ,
    \dest_y[25] , \dest_y[26] , \dest_y[27] , \dest_y[28] , \dest_y[29] ,
    \outport[0] , \outport[1] , \outport[2] , \outport[3] , \outport[4] ,
    \outport[5] , \outport[6] , \outport[7] , \outport[8] , \outport[9] ,
    \outport[10] , \outport[11] , \outport[12] , \outport[13] ,
    \outport[14] , \outport[15] , \outport[16] , \outport[17] ,
    \outport[18] , \outport[19] , \outport[20] , \outport[21] ,
    \outport[22] , \outport[23] , \outport[24] , \outport[25] ,
    \outport[26] , \outport[27] , \outport[28] , \outport[29]   );
  input  \dest_x[0] , \dest_x[1] , \dest_x[2] , \dest_x[3] , \dest_x[4] ,
    \dest_x[5] , \dest_x[6] , \dest_x[7] , \dest_x[8] , \dest_x[9] ,
    \dest_x[10] , \dest_x[11] , \dest_x[12] , \dest_x[13] , \dest_x[14] ,
    \dest_x[15] , \dest_x[16] , \dest_x[17] , \dest_x[18] , \dest_x[19] ,
    \dest_x[20] , \dest_x[21] , \dest_x[22] , \dest_x[23] , \dest_x[24] ,
    \dest_x[25] , \dest_x[26] , \dest_x[27] , \dest_x[28] , \dest_x[29] ,
    \dest_y[0] , \dest_y[1] , \dest_y[2] , \dest_y[3] , \dest_y[4] ,
    \dest_y[5] , \dest_y[6] , \dest_y[7] , \dest_y[8] , \dest_y[9] ,
    \dest_y[10] , \dest_y[11] , \dest_y[12] , \dest_y[13] , \dest_y[14] ,
    \dest_y[15] , \dest_y[16] , \dest_y[17] , \dest_y[18] , \dest_y[19] ,
    \dest_y[20] , \dest_y[21] , \dest_y[22] , \dest_y[23] , \dest_y[24] ,
    \dest_y[25] , \dest_y[26] , \dest_y[27] , \dest_y[28] , \dest_y[29] ;
  output \outport[0] , \outport[1] , \outport[2] , \outport[3] , \outport[4] ,
    \outport[5] , \outport[6] , \outport[7] , \outport[8] , \outport[9] ,
    \outport[10] , \outport[11] , \outport[12] , \outport[13] ,
    \outport[14] , \outport[15] , \outport[16] , \outport[17] ,
    \outport[18] , \outport[19] , \outport[20] , \outport[21] ,
    \outport[22] , \outport[23] , \outport[24] , \outport[25] ,
    \outport[26] , \outport[27] , \outport[28] , \outport[29] ;
  wire new_n91, new_n92, new_n93, new_n94, new_n95, new_n96, new_n97,
    new_n98, new_n99, new_n100, new_n101, new_n102, new_n103, new_n104,
    new_n105, new_n106, new_n107, new_n108, new_n109, new_n110, new_n111,
    new_n112, new_n113, new_n114, new_n115, new_n116, new_n117, new_n118,
    new_n119, new_n120, new_n121, new_n122, new_n123, new_n124, new_n125,
    new_n126, new_n127, new_n128, new_n129, new_n130, new_n131, new_n132,
    new_n133, new_n134, new_n135, new_n136, new_n137, new_n138, new_n139,
    new_n140, new_n141, new_n142, new_n143, new_n144, new_n145, new_n146,
    new_n147, new_n148, new_n149, new_n150, new_n151, new_n152, new_n153,
    new_n154, new_n155, new_n156, new_n157, new_n158, new_n159, new_n160,
    new_n161, new_n162, new_n163, new_n164, new_n165, new_n166, new_n167,
    new_n168, new_n169, new_n170, new_n171, new_n172, new_n173, new_n174,
    new_n175, new_n176, new_n177, new_n178, new_n179, new_n180, new_n181,
    new_n182, new_n183, new_n184, new_n185, new_n186, new_n187, new_n188,
    new_n189, new_n190, new_n191, new_n192, new_n193, new_n194, new_n195,
    new_n196, new_n197, new_n198, new_n199, new_n200, new_n201, new_n202,
    new_n203, new_n204, new_n205, new_n206, new_n207, new_n208, new_n209,
    new_n210, new_n211, new_n212, new_n213, new_n214, new_n216, new_n217,
    new_n218, new_n219, new_n220, new_n221, new_n222, new_n223, new_n224,
    new_n225, new_n226, new_n227, new_n228, new_n229, new_n230, new_n231,
    new_n232, new_n233, new_n234, new_n235, new_n236, new_n237, new_n238,
    new_n239, new_n240, new_n241, new_n242, new_n243, new_n244, new_n245,
    new_n246, new_n247, new_n248, new_n249, new_n250, new_n251, new_n252,
    new_n253, new_n254, new_n255, new_n256, new_n257, new_n258, new_n259,
    new_n260, new_n261, new_n262, new_n263, new_n264, new_n265, new_n266,
    new_n267, new_n268, new_n269, new_n270, new_n271, new_n272, new_n273,
    new_n274, new_n275, new_n276, new_n277, new_n278, new_n279, new_n280,
    new_n281, new_n282, new_n283, new_n284, new_n285, new_n286, new_n287,
    new_n288, new_n289, new_n290, new_n291, new_n292, new_n293, new_n294,
    new_n295, new_n296, new_n297, new_n298, new_n299, new_n300, new_n301,
    new_n302, new_n303, new_n304, new_n305, new_n306, new_n307, new_n308,
    new_n309, new_n310, new_n311, new_n312, new_n313, new_n314, new_n315,
    new_n316, new_n317, new_n318, new_n319, new_n320, new_n321, new_n322,
    new_n323, new_n324, new_n325, new_n326, new_n327, new_n328, new_n329,
    new_n330, new_n331, new_n332, new_n333, new_n334, new_n335, new_n336,
    new_n337, new_n338, new_n339, new_n340, new_n341, new_n342, new_n344,
    new_n345, new_n346;
  assign new_n91 = ~\dest_x[9]  & ~\dest_x[10] ;
  assign new_n92 = \dest_x[9]  & \dest_x[10] ;
  assign new_n93 = ~new_n91 & ~new_n92;
  assign new_n94 = ~new_n91 & \dest_x[11] ;
  assign new_n95 = ~\dest_x[11]  & new_n91;
  assign new_n96 = ~new_n94 & ~new_n95;
  assign new_n97 = ~\dest_x[12]  & ~new_n94;
  assign new_n98 = \dest_x[12]  & new_n94;
  assign new_n99 = ~new_n97 & ~new_n98;
  assign new_n100 = ~\dest_x[13]  & new_n97;
  assign new_n101 = ~new_n97 & \dest_x[13] ;
  assign new_n102 = ~new_n100 & ~new_n101;
  assign new_n103 = ~new_n100 & \dest_x[14] ;
  assign new_n104 = ~\dest_x[14]  & new_n100;
  assign new_n105 = ~new_n103 & ~new_n104;
  assign new_n106 = ~new_n103 & \dest_x[15] ;
  assign new_n107 = ~\dest_x[15]  & new_n103;
  assign new_n108 = ~new_n106 & ~new_n107;
  assign new_n109 = \dest_x[15]  & new_n103;
  assign new_n110 = ~\dest_x[16]  & ~new_n109;
  assign new_n111 = \dest_x[16]  & new_n109;
  assign new_n112 = ~new_n110 & ~new_n111;
  assign new_n113 = ~new_n110 & \dest_x[17] ;
  assign new_n114 = ~\dest_x[17]  & new_n110;
  assign new_n115 = ~new_n113 & ~new_n114;
  assign new_n116 = ~\dest_x[18]  & ~new_n113;
  assign new_n117 = \dest_x[18]  & new_n113;
  assign new_n118 = ~new_n116 & ~new_n117;
  assign new_n119 = ~new_n116 & \dest_x[19] ;
  assign new_n120 = ~\dest_x[19]  & new_n116;
  assign new_n121 = ~new_n119 & ~new_n120;
  assign new_n122 = ~new_n119 & \dest_x[20] ;
  assign new_n123 = ~\dest_x[20]  & new_n119;
  assign new_n124 = ~new_n122 & ~new_n123;
  assign new_n125 = \dest_x[20]  & new_n119;
  assign new_n126 = ~\dest_x[21]  & ~new_n125;
  assign new_n127 = \dest_x[21]  & new_n125;
  assign new_n128 = ~new_n126 & ~new_n127;
  assign new_n129 = ~\dest_x[22]  & new_n126;
  assign new_n130 = ~new_n126 & \dest_x[22] ;
  assign new_n131 = ~new_n129 & ~new_n130;
  assign new_n132 = ~new_n129 & \dest_x[23] ;
  assign new_n133 = ~\dest_x[23]  & new_n129;
  assign new_n134 = ~new_n132 & ~new_n133;
  assign new_n135 = ~new_n132 & \dest_x[24] ;
  assign new_n136 = ~\dest_x[24]  & new_n132;
  assign new_n137 = ~new_n135 & ~new_n136;
  assign new_n138 = \dest_x[24]  & new_n132;
  assign new_n139 = ~new_n138 & \dest_x[25] ;
  assign new_n140 = ~\dest_x[25]  & new_n138;
  assign new_n141 = ~new_n139 & ~new_n140;
  assign new_n142 = \dest_x[25]  & new_n138;
  assign new_n143 = ~\dest_x[26]  & ~new_n142;
  assign new_n144 = \dest_x[26]  & new_n142;
  assign new_n145 = ~new_n143 & ~new_n144;
  assign new_n146 = ~new_n143 & \dest_x[27] ;
  assign new_n147 = ~\dest_x[27]  & new_n143;
  assign new_n148 = ~new_n146 & ~new_n147;
  assign new_n149 = ~new_n146 & \dest_x[28] ;
  assign new_n150 = ~\dest_x[28]  & new_n146;
  assign new_n151 = ~new_n149 & ~new_n150;
  assign new_n152 = \dest_x[28]  & new_n146;
  assign new_n153 = ~\dest_x[29]  & new_n152;
  assign new_n154 = ~new_n152 & \dest_x[29] ;
  assign new_n155 = ~new_n153 & ~new_n154;
  assign new_n156 = ~\dest_x[9]  & ~new_n155;
  assign new_n157 = ~new_n151 & new_n156;
  assign new_n158 = new_n148 & new_n157;
  assign new_n159 = ~new_n145 & new_n158;
  assign new_n160 = ~new_n141 & new_n159;
  assign new_n161 = ~new_n137 & new_n160;
  assign new_n162 = new_n134 & new_n161;
  assign new_n163 = ~new_n131 & new_n162;
  assign new_n164 = ~new_n128 & new_n163;
  assign new_n165 = ~new_n124 & new_n164;
  assign new_n166 = new_n121 & new_n165;
  assign new_n167 = ~new_n118 & new_n166;
  assign new_n168 = new_n115 & new_n167;
  assign new_n169 = ~new_n112 & new_n168;
  assign new_n170 = ~new_n108 & new_n169;
  assign new_n171 = new_n105 & new_n170;
  assign new_n172 = ~new_n102 & new_n171;
  assign new_n173 = ~new_n99 & new_n172;
  assign new_n174 = new_n96 & new_n173;
  assign new_n175 = ~new_n93 & new_n174;
  assign new_n176 = \dest_x[8]  & new_n175;
  assign new_n177 = \dest_x[7]  & new_n176;
  assign new_n178 = \dest_x[6]  & new_n177;
  assign new_n179 = \dest_x[5]  & new_n178;
  assign new_n180 = \dest_x[4]  & new_n179;
  assign new_n181 = \dest_x[3]  & new_n180;
  assign new_n182 = \dest_x[2]  & new_n181;
  assign new_n183 = \dest_x[1]  & new_n182;
  assign new_n184 = \dest_x[0]  & new_n183;
  assign new_n185 = \dest_x[29]  & new_n152;
  assign new_n186 = ~new_n184 & ~new_n185;
  assign new_n187 = ~\dest_x[1]  & ~\dest_x[2] ;
  assign new_n188 = ~\dest_x[3]  & new_n187;
  assign new_n189 = ~\dest_x[4]  & new_n188;
  assign new_n190 = ~\dest_x[5]  & new_n189;
  assign new_n191 = ~\dest_x[6]  & new_n190;
  assign new_n192 = ~\dest_x[7]  & new_n191;
  assign new_n193 = ~\dest_x[8]  & new_n192;
  assign new_n194 = new_n93 & new_n193;
  assign new_n195 = ~new_n96 & new_n194;
  assign new_n196 = new_n99 & new_n195;
  assign new_n197 = new_n102 & new_n196;
  assign new_n198 = ~new_n105 & new_n197;
  assign new_n199 = new_n108 & new_n198;
  assign new_n200 = new_n112 & new_n199;
  assign new_n201 = ~new_n115 & new_n200;
  assign new_n202 = new_n118 & new_n201;
  assign new_n203 = ~new_n121 & new_n202;
  assign new_n204 = new_n124 & new_n203;
  assign new_n205 = new_n128 & new_n204;
  assign new_n206 = new_n131 & new_n205;
  assign new_n207 = ~new_n134 & new_n206;
  assign new_n208 = new_n137 & new_n207;
  assign new_n209 = new_n141 & new_n208;
  assign new_n210 = new_n145 & new_n209;
  assign new_n211 = ~new_n148 & new_n210;
  assign new_n212 = new_n151 & new_n211;
  assign new_n213 = \dest_x[9]  & new_n212;
  assign new_n214 = ~new_n213 & new_n185;
  assign \outport[0]  = new_n186 | new_n214;
  assign new_n216 = ~\dest_y[9]  & ~\dest_y[10] ;
  assign new_n217 = ~new_n216 & \dest_y[11] ;
  assign new_n218 = ~\dest_y[12]  & ~new_n217;
  assign new_n219 = ~\dest_y[13]  & new_n218;
  assign new_n220 = ~new_n219 & \dest_y[14] ;
  assign new_n221 = \dest_y[15]  & new_n220;
  assign new_n222 = ~\dest_y[16]  & ~new_n221;
  assign new_n223 = ~new_n222 & \dest_y[17] ;
  assign new_n224 = ~\dest_y[18]  & ~new_n223;
  assign new_n225 = ~new_n224 & \dest_y[19] ;
  assign new_n226 = \dest_y[20]  & new_n225;
  assign new_n227 = ~\dest_y[21]  & ~new_n226;
  assign new_n228 = ~\dest_y[22]  & new_n227;
  assign new_n229 = ~new_n228 & \dest_y[23] ;
  assign new_n230 = \dest_y[24]  & new_n229;
  assign new_n231 = \dest_y[25]  & new_n230;
  assign new_n232 = ~\dest_y[26]  & ~new_n231;
  assign new_n233 = ~new_n232 & \dest_y[27] ;
  assign new_n234 = \dest_y[28]  & new_n233;
  assign new_n235 = \dest_y[29]  & new_n234;
  assign new_n236 = ~new_n235 & \dest_x[0] ;
  assign new_n237 = ~\dest_x[0]  & ~\dest_y[0] ;
  assign new_n238 = ~new_n237 & new_n235;
  assign new_n239 = \dest_y[9]  & \dest_y[10] ;
  assign new_n240 = ~new_n216 & ~new_n239;
  assign new_n241 = ~\dest_y[11]  & new_n216;
  assign new_n242 = ~new_n217 & ~new_n241;
  assign new_n243 = \dest_y[12]  & new_n217;
  assign new_n244 = ~new_n218 & ~new_n243;
  assign new_n245 = ~new_n218 & \dest_y[13] ;
  assign new_n246 = ~new_n219 & ~new_n245;
  assign new_n247 = ~\dest_y[14]  & new_n219;
  assign new_n248 = ~new_n220 & ~new_n247;
  assign new_n249 = ~new_n220 & \dest_y[15] ;
  assign new_n250 = ~\dest_y[15]  & new_n220;
  assign new_n251 = ~new_n249 & ~new_n250;
  assign new_n252 = \dest_y[16]  & new_n221;
  assign new_n253 = ~new_n222 & ~new_n252;
  assign new_n254 = ~\dest_y[17]  & new_n222;
  assign new_n255 = ~new_n223 & ~new_n254;
  assign new_n256 = \dest_y[18]  & new_n223;
  assign new_n257 = ~new_n224 & ~new_n256;
  assign new_n258 = ~\dest_y[19]  & new_n224;
  assign new_n259 = ~new_n225 & ~new_n258;
  assign new_n260 = ~new_n225 & \dest_y[20] ;
  assign new_n261 = ~\dest_y[20]  & new_n225;
  assign new_n262 = ~new_n260 & ~new_n261;
  assign new_n263 = \dest_y[21]  & new_n226;
  assign new_n264 = ~new_n227 & ~new_n263;
  assign new_n265 = ~new_n227 & \dest_y[22] ;
  assign new_n266 = ~new_n228 & ~new_n265;
  assign new_n267 = ~\dest_y[23]  & new_n228;
  assign new_n268 = ~new_n229 & ~new_n267;
  assign new_n269 = ~new_n229 & \dest_y[24] ;
  assign new_n270 = ~\dest_y[24]  & new_n229;
  assign new_n271 = ~new_n269 & ~new_n270;
  assign new_n272 = ~new_n230 & \dest_y[25] ;
  assign new_n273 = ~\dest_y[25]  & new_n230;
  assign new_n274 = ~new_n272 & ~new_n273;
  assign new_n275 = \dest_y[26]  & new_n231;
  assign new_n276 = ~new_n232 & ~new_n275;
  assign new_n277 = ~\dest_y[27]  & new_n232;
  assign new_n278 = ~new_n233 & ~new_n277;
  assign new_n279 = ~new_n233 & \dest_y[28] ;
  assign new_n280 = ~\dest_y[28]  & new_n233;
  assign new_n281 = ~new_n279 & ~new_n280;
  assign new_n282 = ~\dest_y[9]  & \dest_y[0] ;
  assign new_n283 = \dest_y[29]  & new_n282;
  assign new_n284 = ~new_n281 & new_n283;
  assign new_n285 = new_n278 & new_n284;
  assign new_n286 = ~new_n276 & new_n285;
  assign new_n287 = ~new_n274 & new_n286;
  assign new_n288 = ~new_n271 & new_n287;
  assign new_n289 = new_n268 & new_n288;
  assign new_n290 = ~new_n266 & new_n289;
  assign new_n291 = ~new_n264 & new_n290;
  assign new_n292 = ~new_n262 & new_n291;
  assign new_n293 = new_n259 & new_n292;
  assign new_n294 = ~new_n257 & new_n293;
  assign new_n295 = new_n255 & new_n294;
  assign new_n296 = ~new_n253 & new_n295;
  assign new_n297 = ~new_n251 & new_n296;
  assign new_n298 = new_n248 & new_n297;
  assign new_n299 = ~new_n246 & new_n298;
  assign new_n300 = ~new_n244 & new_n299;
  assign new_n301 = new_n242 & new_n300;
  assign new_n302 = ~new_n240 & new_n301;
  assign new_n303 = \dest_y[8]  & new_n302;
  assign new_n304 = \dest_y[7]  & new_n303;
  assign new_n305 = \dest_y[6]  & new_n304;
  assign new_n306 = \dest_y[5]  & new_n305;
  assign new_n307 = \dest_y[4]  & new_n306;
  assign new_n308 = \dest_y[3]  & new_n307;
  assign new_n309 = \dest_y[2]  & new_n308;
  assign new_n310 = \dest_y[1]  & new_n309;
  assign new_n311 = ~\dest_y[1]  & ~\dest_y[2] ;
  assign new_n312 = ~\dest_y[3]  & new_n311;
  assign new_n313 = ~\dest_y[4]  & new_n312;
  assign new_n314 = ~\dest_y[5]  & new_n313;
  assign new_n315 = ~\dest_y[6]  & new_n314;
  assign new_n316 = ~\dest_y[7]  & new_n315;
  assign new_n317 = ~\dest_y[8]  & new_n316;
  assign new_n318 = new_n240 & new_n317;
  assign new_n319 = ~new_n242 & new_n318;
  assign new_n320 = new_n244 & new_n319;
  assign new_n321 = new_n246 & new_n320;
  assign new_n322 = ~new_n248 & new_n321;
  assign new_n323 = new_n251 & new_n322;
  assign new_n324 = new_n253 & new_n323;
  assign new_n325 = ~new_n255 & new_n324;
  assign new_n326 = new_n257 & new_n325;
  assign new_n327 = ~new_n259 & new_n326;
  assign new_n328 = new_n262 & new_n327;
  assign new_n329 = new_n264 & new_n328;
  assign new_n330 = new_n266 & new_n329;
  assign new_n331 = ~new_n268 & new_n330;
  assign new_n332 = new_n271 & new_n331;
  assign new_n333 = new_n274 & new_n332;
  assign new_n334 = new_n276 & new_n333;
  assign new_n335 = ~new_n278 & new_n334;
  assign new_n336 = new_n281 & new_n335;
  assign new_n337 = \dest_y[9]  & new_n336;
  assign new_n338 = ~new_n337 & new_n235;
  assign new_n339 = ~new_n310 & ~new_n338;
  assign new_n340 = ~new_n238 & new_n339;
  assign new_n341 = ~new_n186 & ~new_n340;
  assign new_n342 = ~new_n236 & new_n341;
  assign \outport[1]  = ~new_n214 & ~new_n342;
  assign new_n344 = \dest_x[0]  & new_n235;
  assign new_n345 = \dest_y[0]  & new_n344;
  assign new_n346 = ~new_n338 & ~new_n345;
  assign \outport[2]  = ~\outport[0]  & ~new_n346;
  assign \outport[3]  = 1'b0;
  assign \outport[4]  = 1'b0;
  assign \outport[5]  = 1'b0;
  assign \outport[6]  = 1'b0;
  assign \outport[7]  = 1'b0;
  assign \outport[8]  = 1'b0;
  assign \outport[9]  = 1'b0;
  assign \outport[10]  = 1'b0;
  assign \outport[11]  = 1'b0;
  assign \outport[12]  = 1'b0;
  assign \outport[13]  = 1'b0;
  assign \outport[14]  = 1'b0;
  assign \outport[15]  = 1'b0;
  assign \outport[16]  = 1'b0;
  assign \outport[17]  = 1'b0;
  assign \outport[18]  = 1'b0;
  assign \outport[19]  = 1'b0;
  assign \outport[20]  = 1'b0;
  assign \outport[21]  = 1'b0;
  assign \outport[22]  = 1'b0;
  assign \outport[23]  = 1'b0;
  assign \outport[24]  = 1'b0;
  assign \outport[25]  = 1'b0;
  assign \outport[26]  = 1'b0;
  assign \outport[27]  = 1'b0;
  assign \outport[28]  = 1'b0;
  assign \outport[29]  = 1'b0;
endmodule


