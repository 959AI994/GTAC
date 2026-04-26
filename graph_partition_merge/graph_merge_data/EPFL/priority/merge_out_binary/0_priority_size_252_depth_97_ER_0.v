// Benchmark "priority" written by ABC on Thu Apr  2 15:01:23 2026

module priority ( 
    \A[0] , \A[1] , \A[2] , \A[3] , \A[4] , \A[5] , \A[6] , \A[7] , \A[8] ,
    \A[9] , \A[10] , \A[11] , \A[12] , \A[13] , \A[14] , \A[15] , \A[16] ,
    \A[17] , \A[18] , \A[19] , \A[20] , \A[21] , \A[22] , \A[23] , \A[24] ,
    \A[25] , \A[26] , \A[27] , \A[28] , \A[29] , \A[30] , \A[31] , \A[32] ,
    \A[33] , \A[34] , \A[35] , \A[36] , \A[37] , \A[38] , \A[39] , \A[40] ,
    \A[41] , \A[42] , \A[43] , \A[44] , \A[45] , \A[46] , \A[47] , \A[48] ,
    \A[49] , \A[50] , \A[51] , \A[52] , \A[53] , \A[54] , \A[55] , \A[56] ,
    \A[57] , \A[58] , \A[59] , \A[60] , \A[61] , \A[62] , \A[63] , \A[64] ,
    \A[65] , \A[66] , \A[67] , \A[68] , \A[69] , \A[70] , \A[71] , \A[72] ,
    \A[73] , \A[74] , \A[75] , \A[76] , \A[77] , \A[78] , \A[79] , \A[80] ,
    \A[81] , \A[82] , \A[83] , \A[84] , \A[85] , \A[86] , \A[87] , \A[88] ,
    \A[89] , \A[90] , \A[91] , \A[92] , \A[93] , \A[94] , \A[95] , \A[96] ,
    \A[97] , \A[98] , \A[99] , \A[100] , \A[101] , \A[102] , \A[103] ,
    \A[104] , \A[105] , \A[106] , \A[107] , \A[108] , \A[109] , \A[110] ,
    \A[111] , \A[112] , \A[113] , \A[114] , \A[115] , \A[116] , \A[117] ,
    \A[118] , \A[119] , \A[120] , \A[121] , \A[122] , \A[123] , \A[124] ,
    \A[125] , \A[126] , \A[127] ,
    \P[0] , \P[1] , \P[2] , \P[3] , \P[4] , \P[5] , \P[6] , F  );
  input  \A[0] , \A[1] , \A[2] , \A[3] , \A[4] , \A[5] , \A[6] , \A[7] ,
    \A[8] , \A[9] , \A[10] , \A[11] , \A[12] , \A[13] , \A[14] , \A[15] ,
    \A[16] , \A[17] , \A[18] , \A[19] , \A[20] , \A[21] , \A[22] , \A[23] ,
    \A[24] , \A[25] , \A[26] , \A[27] , \A[28] , \A[29] , \A[30] , \A[31] ,
    \A[32] , \A[33] , \A[34] , \A[35] , \A[36] , \A[37] , \A[38] , \A[39] ,
    \A[40] , \A[41] , \A[42] , \A[43] , \A[44] , \A[45] , \A[46] , \A[47] ,
    \A[48] , \A[49] , \A[50] , \A[51] , \A[52] , \A[53] , \A[54] , \A[55] ,
    \A[56] , \A[57] , \A[58] , \A[59] , \A[60] , \A[61] , \A[62] , \A[63] ,
    \A[64] , \A[65] , \A[66] , \A[67] , \A[68] , \A[69] , \A[70] , \A[71] ,
    \A[72] , \A[73] , \A[74] , \A[75] , \A[76] , \A[77] , \A[78] , \A[79] ,
    \A[80] , \A[81] , \A[82] , \A[83] , \A[84] , \A[85] , \A[86] , \A[87] ,
    \A[88] , \A[89] , \A[90] , \A[91] , \A[92] , \A[93] , \A[94] , \A[95] ,
    \A[96] , \A[97] , \A[98] , \A[99] , \A[100] , \A[101] , \A[102] ,
    \A[103] , \A[104] , \A[105] , \A[106] , \A[107] , \A[108] , \A[109] ,
    \A[110] , \A[111] , \A[112] , \A[113] , \A[114] , \A[115] , \A[116] ,
    \A[117] , \A[118] , \A[119] , \A[120] , \A[121] , \A[122] , \A[123] ,
    \A[124] , \A[125] , \A[126] , \A[127] ;
  output \P[0] , \P[1] , \P[2] , \P[3] , \P[4] , \P[5] , \P[6] , F;
  wire new_n137, new_n138, new_n139, new_n140, new_n141, new_n142, new_n143,
    new_n144, new_n145, new_n146, new_n147, new_n148, new_n149, new_n150,
    new_n151, new_n152, new_n153, new_n154, new_n155, new_n156, new_n157,
    new_n158, new_n159, new_n160, new_n161, new_n162, new_n163, new_n164,
    new_n165, new_n166, new_n167, new_n168, new_n169, new_n170, new_n171,
    new_n172, new_n173, new_n174, new_n175, new_n176, new_n177, new_n178,
    new_n179, new_n180, new_n181, new_n182, new_n183, new_n184, new_n185,
    new_n186, new_n187, new_n188, new_n189, new_n190, new_n191, new_n192,
    new_n193, new_n194, new_n195, new_n196, new_n197, new_n198, new_n199,
    new_n200, new_n201, new_n202, new_n203, new_n204, new_n205, new_n206,
    new_n207, new_n208, new_n209, new_n210, new_n211, new_n212, new_n213,
    new_n214, new_n215, new_n216, new_n217, new_n218, new_n219, new_n220,
    new_n221, new_n222, new_n223, new_n224, new_n225, new_n226, new_n227,
    new_n228, new_n229, new_n230, new_n231, new_n232, new_n233, new_n234,
    new_n235, new_n236, new_n237, new_n238, new_n239, new_n240, new_n241,
    new_n242, new_n243, new_n244, new_n245, new_n246, new_n247, new_n248,
    new_n249, new_n250, new_n251, new_n252, new_n253, new_n254, new_n255,
    new_n256, new_n257, new_n258, new_n259, new_n260, new_n261, new_n262,
    new_n263, new_n264, new_n265, new_n266, new_n267, new_n268, new_n269,
    new_n270, new_n271, new_n272, new_n273, new_n274, new_n275, new_n276,
    new_n277, new_n278, new_n279, new_n280, new_n281, new_n282, new_n283,
    new_n284, new_n285, new_n286, new_n287, new_n288, new_n289, new_n290,
    new_n291, new_n292, new_n293, new_n294, new_n295, new_n296, new_n297,
    new_n298, new_n299, new_n300, new_n301, new_n302, new_n303, new_n304,
    new_n305, new_n306, new_n307, new_n309, new_n310, new_n311, new_n312,
    new_n313, new_n314, new_n315, new_n316, new_n317, new_n318, new_n319,
    new_n320, new_n321, new_n322, new_n323, new_n324, new_n325, new_n326,
    new_n327, new_n328, new_n329, new_n330, new_n331, new_n332, new_n333,
    new_n334, new_n335, new_n336, new_n337, new_n338, new_n339, new_n340,
    new_n341, new_n342, new_n343, new_n344, new_n345, new_n346, new_n347,
    new_n348, new_n349, new_n350, new_n351, new_n353, new_n354, new_n355,
    new_n356, new_n357, new_n358, new_n359, new_n360, new_n361, new_n362,
    new_n363, new_n364, new_n365, new_n366, new_n367, new_n368, new_n370,
    new_n371, new_n372, new_n373, new_n374, new_n375, new_n376, new_n377,
    new_n378, new_n379, new_n380, new_n381, new_n382, new_n383, new_n384,
    new_n385, new_n386;
  assign new_n137 = ~\A[119]  & \A[118] ;
  assign new_n138 = ~\A[120]  & ~new_n137;
  assign new_n139 = ~\A[121]  & ~new_n138;
  assign new_n140 = ~\A[122]  & ~new_n139;
  assign new_n141 = ~\A[123]  & ~new_n140;
  assign new_n142 = ~\A[124]  & ~new_n141;
  assign new_n143 = ~\A[125]  & ~new_n142;
  assign new_n144 = ~\A[126]  & ~new_n143;
  assign new_n145 = ~\A[127]  & ~new_n144;
  assign new_n146 = ~\A[116]  & ~new_n145;
  assign new_n147 = \A[117]  & new_n144;
  assign new_n148 = ~\A[120]  & \A[119] ;
  assign new_n149 = ~\A[121]  & ~new_n148;
  assign new_n150 = ~\A[122]  & ~new_n149;
  assign new_n151 = ~\A[123]  & ~new_n150;
  assign new_n152 = ~\A[124]  & ~new_n151;
  assign new_n153 = ~\A[125]  & ~new_n152;
  assign new_n154 = ~\A[117]  & ~\A[126] ;
  assign new_n155 = ~new_n153 & new_n154;
  assign new_n156 = ~\A[127]  & ~new_n155;
  assign new_n157 = ~new_n147 & new_n156;
  assign new_n158 = ~new_n157 & \A[116] ;
  assign new_n159 = ~new_n146 & ~new_n158;
  assign new_n160 = ~\A[114]  & ~new_n159;
  assign new_n161 = ~\A[115]  & ~new_n157;
  assign new_n162 = ~new_n159 & \A[115] ;
  assign new_n163 = ~new_n161 & ~new_n162;
  assign new_n164 = ~new_n163 & \A[114] ;
  assign new_n165 = ~new_n160 & ~new_n164;
  assign new_n166 = ~\A[112]  & ~new_n165;
  assign new_n167 = ~\A[113]  & ~new_n163;
  assign new_n168 = ~new_n165 & \A[113] ;
  assign new_n169 = ~new_n167 & ~new_n168;
  assign new_n170 = ~new_n169 & \A[112] ;
  assign new_n171 = ~new_n166 & ~new_n170;
  assign new_n172 = ~\A[110]  & ~new_n171;
  assign new_n173 = ~\A[111]  & ~new_n169;
  assign new_n174 = ~new_n171 & \A[111] ;
  assign new_n175 = ~new_n173 & ~new_n174;
  assign new_n176 = ~new_n175 & \A[110] ;
  assign new_n177 = ~new_n172 & ~new_n176;
  assign new_n178 = ~\A[108]  & ~new_n177;
  assign new_n179 = ~\A[109]  & ~new_n175;
  assign new_n180 = ~new_n177 & \A[109] ;
  assign new_n181 = ~new_n179 & ~new_n180;
  assign new_n182 = ~new_n181 & \A[108] ;
  assign new_n183 = ~new_n178 & ~new_n182;
  assign new_n184 = ~\A[106]  & ~new_n183;
  assign new_n185 = ~\A[107]  & ~new_n181;
  assign new_n186 = ~new_n183 & \A[107] ;
  assign new_n187 = ~new_n185 & ~new_n186;
  assign new_n188 = ~new_n187 & \A[106] ;
  assign new_n189 = ~new_n184 & ~new_n188;
  assign new_n190 = ~\A[104]  & ~new_n189;
  assign new_n191 = ~\A[105]  & ~new_n187;
  assign new_n192 = ~new_n189 & \A[105] ;
  assign new_n193 = ~new_n191 & ~new_n192;
  assign new_n194 = ~new_n193 & \A[104] ;
  assign new_n195 = ~new_n190 & ~new_n194;
  assign new_n196 = ~\A[102]  & ~new_n195;
  assign new_n197 = ~\A[103]  & ~new_n193;
  assign new_n198 = ~new_n195 & \A[103] ;
  assign new_n199 = ~new_n197 & ~new_n198;
  assign new_n200 = ~new_n199 & \A[102] ;
  assign new_n201 = ~new_n196 & ~new_n200;
  assign new_n202 = ~\A[100]  & ~new_n201;
  assign new_n203 = ~\A[101]  & ~new_n199;
  assign new_n204 = ~new_n201 & \A[101] ;
  assign new_n205 = ~new_n203 & ~new_n204;
  assign new_n206 = ~new_n205 & \A[100] ;
  assign new_n207 = ~new_n202 & ~new_n206;
  assign new_n208 = ~\A[98]  & ~new_n207;
  assign new_n209 = ~\A[99]  & ~new_n205;
  assign new_n210 = ~new_n207 & \A[99] ;
  assign new_n211 = ~new_n209 & ~new_n210;
  assign new_n212 = ~new_n211 & \A[98] ;
  assign new_n213 = ~new_n208 & ~new_n212;
  assign new_n214 = ~\A[96]  & ~new_n213;
  assign new_n215 = ~\A[97]  & ~new_n211;
  assign new_n216 = ~new_n213 & \A[97] ;
  assign new_n217 = ~new_n215 & ~new_n216;
  assign new_n218 = ~new_n217 & \A[96] ;
  assign new_n219 = ~new_n214 & ~new_n218;
  assign new_n220 = ~\A[94]  & ~new_n219;
  assign new_n221 = ~\A[95]  & ~new_n217;
  assign new_n222 = ~new_n219 & \A[95] ;
  assign new_n223 = ~new_n221 & ~new_n222;
  assign new_n224 = ~new_n223 & \A[94] ;
  assign new_n225 = ~new_n220 & ~new_n224;
  assign new_n226 = ~\A[92]  & ~new_n225;
  assign new_n227 = ~\A[93]  & ~new_n223;
  assign new_n228 = ~new_n225 & \A[93] ;
  assign new_n229 = ~new_n227 & ~new_n228;
  assign new_n230 = ~new_n229 & \A[92] ;
  assign new_n231 = ~new_n226 & ~new_n230;
  assign new_n232 = ~\A[90]  & ~new_n231;
  assign new_n233 = ~\A[91]  & ~new_n229;
  assign new_n234 = ~new_n231 & \A[91] ;
  assign new_n235 = ~new_n233 & ~new_n234;
  assign new_n236 = ~new_n235 & \A[90] ;
  assign new_n237 = ~new_n232 & ~new_n236;
  assign new_n238 = ~\A[88]  & ~new_n237;
  assign new_n239 = ~\A[89]  & ~new_n235;
  assign new_n240 = ~new_n237 & \A[89] ;
  assign new_n241 = ~new_n239 & ~new_n240;
  assign new_n242 = ~new_n241 & \A[88] ;
  assign new_n243 = ~new_n238 & ~new_n242;
  assign new_n244 = ~\A[86]  & ~new_n243;
  assign new_n245 = ~\A[87]  & ~new_n241;
  assign new_n246 = ~new_n243 & \A[87] ;
  assign new_n247 = ~new_n245 & ~new_n246;
  assign new_n248 = ~new_n247 & \A[86] ;
  assign new_n249 = ~new_n244 & ~new_n248;
  assign new_n250 = ~\A[84]  & ~new_n249;
  assign new_n251 = ~\A[85]  & ~new_n247;
  assign new_n252 = ~new_n249 & \A[85] ;
  assign new_n253 = ~new_n251 & ~new_n252;
  assign new_n254 = ~new_n253 & \A[84] ;
  assign new_n255 = ~new_n250 & ~new_n254;
  assign new_n256 = ~\A[82]  & ~new_n255;
  assign new_n257 = ~\A[83]  & ~new_n253;
  assign new_n258 = ~new_n255 & \A[83] ;
  assign new_n259 = ~new_n257 & ~new_n258;
  assign new_n260 = ~new_n259 & \A[82] ;
  assign new_n261 = ~new_n256 & ~new_n260;
  assign new_n262 = ~\A[80]  & ~new_n261;
  assign new_n263 = ~\A[81]  & ~new_n259;
  assign new_n264 = ~new_n261 & \A[81] ;
  assign new_n265 = ~new_n263 & ~new_n264;
  assign new_n266 = ~new_n265 & \A[80] ;
  assign new_n267 = ~new_n262 & ~new_n266;
  assign new_n268 = ~\A[78]  & ~new_n267;
  assign new_n269 = ~\A[79]  & ~new_n265;
  assign new_n270 = ~new_n267 & \A[79] ;
  assign new_n271 = ~new_n269 & ~new_n270;
  assign new_n272 = ~new_n271 & \A[78] ;
  assign new_n273 = ~new_n268 & ~new_n272;
  assign new_n274 = ~\A[76]  & ~new_n273;
  assign new_n275 = ~\A[77]  & ~new_n271;
  assign new_n276 = ~new_n273 & \A[77] ;
  assign new_n277 = ~new_n275 & ~new_n276;
  assign new_n278 = ~new_n277 & \A[76] ;
  assign new_n279 = ~new_n274 & ~new_n278;
  assign new_n280 = ~\A[74]  & ~new_n279;
  assign new_n281 = ~\A[75]  & ~new_n277;
  assign new_n282 = ~new_n279 & \A[75] ;
  assign new_n283 = ~new_n281 & ~new_n282;
  assign new_n284 = ~new_n283 & \A[74] ;
  assign new_n285 = ~\A[54]  & \A[53] ;
  assign new_n286 = ~\A[55]  & ~new_n285;
  assign new_n287 = ~\A[56]  & ~new_n286;
  assign new_n288 = ~\A[57]  & ~new_n287;
  assign new_n289 = ~\A[58]  & ~new_n288;
  assign new_n290 = ~\A[59]  & ~new_n289;
  assign new_n291 = ~\A[60]  & ~new_n290;
  assign new_n292 = ~\A[61]  & ~new_n291;
  assign new_n293 = ~\A[62]  & ~new_n292;
  assign new_n294 = ~\A[63]  & ~new_n293;
  assign new_n295 = ~\A[64]  & ~new_n294;
  assign new_n296 = ~\A[65]  & ~new_n295;
  assign new_n297 = ~\A[66]  & ~new_n296;
  assign new_n298 = ~\A[67]  & ~new_n297;
  assign new_n299 = ~\A[68]  & ~new_n298;
  assign new_n300 = ~\A[69]  & ~new_n299;
  assign new_n301 = ~\A[70]  & ~new_n300;
  assign new_n302 = ~\A[71]  & ~new_n301;
  assign new_n303 = ~\A[72]  & ~new_n302;
  assign new_n304 = ~\A[73]  & ~new_n303;
  assign new_n305 = ~new_n280 & ~new_n304;
  assign new_n306 = ~new_n284 & new_n305;
  assign new_n307 = new_n283 & new_n304;
  assign \P[0]  = ~new_n306 & ~new_n307;
  assign new_n309 = ~\A[126]  & ~\A[127] ;
  assign new_n310 = ~\A[124]  & ~\A[125] ;
  assign new_n311 = ~\A[122]  & ~\A[123] ;
  assign new_n312 = ~\A[120]  & ~\A[121] ;
  assign new_n313 = ~\A[118]  & ~\A[119] ;
  assign new_n314 = ~\A[116]  & ~\A[117] ;
  assign new_n315 = ~\A[114]  & ~\A[115] ;
  assign new_n316 = ~\A[112]  & ~\A[113] ;
  assign new_n317 = ~\A[110]  & ~\A[111] ;
  assign new_n318 = ~\A[108]  & ~\A[109] ;
  assign new_n319 = ~new_n318 & new_n317;
  assign new_n320 = ~new_n319 & new_n316;
  assign new_n321 = ~new_n320 & new_n315;
  assign new_n322 = ~\A[106]  & ~\A[107] ;
  assign new_n323 = ~\A[104]  & ~\A[105] ;
  assign new_n324 = ~\A[102]  & ~\A[103] ;
  assign new_n325 = ~\A[100]  & ~\A[101] ;
  assign new_n326 = ~\A[98]  & ~\A[99] ;
  assign new_n327 = ~\A[96]  & ~\A[97] ;
  assign new_n328 = ~\A[94]  & ~\A[95] ;
  assign new_n329 = ~\A[92]  & ~\A[93] ;
  assign new_n330 = ~\A[88]  & ~\A[89] ;
  assign new_n331 = ~\A[90]  & ~\A[91] ;
  assign new_n332 = ~new_n330 & new_n331;
  assign new_n333 = ~new_n332 & new_n329;
  assign new_n334 = ~new_n333 & new_n328;
  assign new_n335 = ~new_n334 & new_n327;
  assign new_n336 = ~new_n335 & new_n326;
  assign new_n337 = ~new_n336 & new_n325;
  assign new_n338 = ~new_n337 & new_n324;
  assign new_n339 = ~new_n338 & new_n323;
  assign new_n340 = ~new_n339 & new_n322;
  assign new_n341 = ~new_n321 & new_n314;
  assign new_n342 = ~new_n340 & new_n341;
  assign new_n343 = ~new_n317 & new_n316;
  assign new_n344 = ~new_n343 & new_n315;
  assign new_n345 = ~new_n344 & new_n314;
  assign new_n346 = new_n340 & new_n345;
  assign new_n347 = ~new_n342 & new_n313;
  assign new_n348 = ~new_n346 & new_n347;
  assign new_n349 = ~new_n348 & new_n312;
  assign new_n350 = ~new_n349 & new_n311;
  assign new_n351 = ~new_n350 & new_n310;
  assign \P[1]  = new_n351 | ~new_n309;
  assign new_n353 = new_n309 & new_n310;
  assign new_n354 = new_n311 & new_n312;
  assign new_n355 = new_n313 & new_n314;
  assign new_n356 = new_n317 & new_n318;
  assign new_n357 = new_n322 & new_n323;
  assign new_n358 = new_n328 & new_n329;
  assign new_n359 = new_n326 & new_n327;
  assign new_n360 = ~new_n358 & new_n359;
  assign new_n361 = new_n324 & new_n325;
  assign new_n362 = ~new_n360 & new_n361;
  assign new_n363 = ~new_n362 & new_n357;
  assign new_n364 = ~new_n363 & new_n356;
  assign new_n365 = new_n315 & new_n316;
  assign new_n366 = ~new_n364 & new_n365;
  assign new_n367 = ~new_n366 & new_n355;
  assign new_n368 = ~new_n367 & new_n354;
  assign \P[2]  = new_n368 | ~new_n353;
  assign new_n370 = new_n353 & new_n354;
  assign new_n371 = new_n355 & new_n365;
  assign new_n372 = ~\A[80]  & ~\A[81] ;
  assign new_n373 = ~\A[82]  & ~\A[83] ;
  assign new_n374 = ~\A[84]  & ~\A[85] ;
  assign new_n375 = ~\A[86]  & ~\A[87] ;
  assign new_n376 = new_n374 & new_n375;
  assign new_n377 = new_n372 & new_n373;
  assign new_n378 = new_n376 & new_n377;
  assign new_n379 = new_n330 & new_n331;
  assign new_n380 = new_n358 & new_n379;
  assign new_n381 = ~new_n378 & new_n380;
  assign new_n382 = new_n359 & new_n361;
  assign new_n383 = ~new_n381 & new_n382;
  assign new_n384 = new_n356 & new_n357;
  assign new_n385 = ~new_n383 & new_n384;
  assign new_n386 = ~new_n385 & new_n371;
  assign \P[3]  = new_n386 | ~new_n370;
  assign \P[4]  = ~new_n370 | ~new_n371;
  assign \P[5]  = 1'b1;
  assign \P[6]  = 1'b1;
  assign F = 1'b1;
endmodule


