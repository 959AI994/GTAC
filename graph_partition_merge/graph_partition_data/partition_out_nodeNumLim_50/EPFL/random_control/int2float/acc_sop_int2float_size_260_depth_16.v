// Benchmark "../EPFL/benchmarks/random_control/int2float" written by ABC on Mon Nov  3 15:57:07 2025

module int2float  ( 
    \B[0] , \B[1] , \B[2] , \B[3] , \B[4] , \B[5] , \B[6] , \B[7] , \B[8] ,
    \B[9] , \B[10] ,
    \M[0] , \M[1] , \M[2] , \M[3] , \E[0] , \E[1] , \E[2]   );
  input  \B[0] , \B[1] , \B[2] , \B[3] , \B[4] , \B[5] , \B[6] , \B[7] ,
    \B[8] , \B[9] , \B[10] ;
  output \M[0] , \M[1] , \M[2] , \M[3] , \E[0] , \E[1] , \E[2] ;
  wire new_n19, new_n20, new_n21, new_n22, new_n23, new_n24, new_n25,
    new_n26, new_n27, new_n28, new_n29, new_n30, new_n31, new_n32, new_n33,
    new_n34, new_n35, new_n36, new_n37, new_n38, new_n39, new_n40, new_n41,
    new_n42, new_n43, new_n44, new_n45, new_n46, new_n47, new_n48, new_n49,
    new_n50, new_n51, new_n52, new_n53, new_n54, new_n55, new_n56, new_n57,
    new_n58, new_n59, new_n60, new_n61, new_n62, new_n63, new_n64, new_n65,
    new_n66, new_n67, new_n68, new_n69, new_n70, new_n71, new_n72, new_n73,
    new_n74, new_n75, new_n76, new_n77, new_n79, new_n80, new_n81, new_n82,
    new_n83, new_n84, new_n85, new_n86, new_n87, new_n88, new_n89, new_n90,
    new_n91, new_n92, new_n93, new_n94, new_n95, new_n96, new_n97, new_n98,
    new_n99, new_n100, new_n101, new_n102, new_n103, new_n104, new_n105,
    new_n106, new_n107, new_n108, new_n109, new_n110, new_n111, new_n112,
    new_n113, new_n114, new_n115, new_n116, new_n117, new_n118, new_n119,
    new_n120, new_n121, new_n122, new_n123, new_n124, new_n125, new_n126,
    new_n127, new_n128, new_n129, new_n130, new_n131, new_n132, new_n133,
    new_n134, new_n135, new_n136, new_n137, new_n138, new_n139, new_n140,
    new_n141, new_n142, new_n143, new_n144, new_n145, new_n147, new_n148,
    new_n149, new_n150, new_n151, new_n152, new_n153, new_n154, new_n155,
    new_n156, new_n157, new_n158, new_n159, new_n160, new_n161, new_n162,
    new_n163, new_n164, new_n165, new_n166, new_n167, new_n168, new_n169,
    new_n170, new_n171, new_n172, new_n173, new_n174, new_n175, new_n176,
    new_n177, new_n178, new_n179, new_n180, new_n181, new_n182, new_n183,
    new_n184, new_n185, new_n186, new_n187, new_n188, new_n189, new_n190,
    new_n191, new_n192, new_n193, new_n194, new_n195, new_n196, new_n197,
    new_n198, new_n199, new_n200, new_n201, new_n202, new_n203, new_n204,
    new_n205, new_n206, new_n207, new_n208, new_n210, new_n211, new_n212,
    new_n213, new_n214, new_n215, new_n216, new_n217, new_n218, new_n220,
    new_n221, new_n222, new_n223, new_n224, new_n225, new_n226, new_n227,
    new_n228, new_n229, new_n230, new_n231, new_n232, new_n233, new_n234,
    new_n235, new_n236, new_n237, new_n238, new_n239, new_n240, new_n241,
    new_n242, new_n243, new_n244, new_n245, new_n246, new_n247, new_n248,
    new_n249, new_n250, new_n252, new_n253, new_n254, new_n255, new_n256,
    new_n257, new_n258, new_n259, new_n260, new_n261, new_n262, new_n263,
    new_n264, new_n265, new_n266, new_n267, new_n268, new_n269, new_n270,
    new_n271, new_n272, new_n274, new_n275, new_n276, new_n277;
  assign new_n19 = ~\B[1]  & \B[4] ;
  assign new_n20 = ~\B[4]  & ~\B[8] ;
  assign new_n21 = ~new_n19 & ~new_n20;
  assign new_n22 = ~new_n21 & \B[0] ;
  assign new_n23 = \B[1]  & \B[4] ;
  assign new_n24 = ~\B[0]  & new_n23;
  assign new_n25 = ~new_n22 & ~new_n24;
  assign new_n26 = ~\B[6]  & ~new_n25;
  assign new_n27 = ~\B[7]  & new_n26;
  assign new_n28 = \B[4]  & \B[8] ;
  assign new_n29 = ~new_n27 & ~new_n28;
  assign new_n30 = ~\B[5]  & ~new_n29;
  assign new_n31 = ~\B[4]  & \B[7] ;
  assign new_n32 = ~\B[2]  & \B[1] ;
  assign new_n33 = ~\B[7]  & \B[5] ;
  assign new_n34 = new_n32 & new_n33;
  assign new_n35 = ~new_n31 & ~new_n34;
  assign new_n36 = ~new_n35 & \B[3] ;
  assign new_n37 = \B[4]  & \B[7] ;
  assign new_n38 = ~\B[3]  & new_n37;
  assign new_n39 = ~new_n36 & ~new_n38;
  assign new_n40 = ~\B[8]  & ~new_n39;
  assign new_n41 = \B[5]  & \B[8] ;
  assign new_n42 = ~\B[4]  & new_n41;
  assign new_n43 = ~new_n40 & ~new_n42;
  assign new_n44 = ~new_n30 & new_n43;
  assign new_n45 = ~\B[9]  & ~new_n44;
  assign new_n46 = ~\B[8]  & \B[4] ;
  assign new_n47 = ~\B[3]  & new_n46;
  assign new_n48 = ~\B[4]  & ~\B[7] ;
  assign new_n49 = ~new_n47 & ~new_n48;
  assign new_n50 = ~\B[2]  & ~new_n49;
  assign new_n51 = \B[1]  & new_n50;
  assign new_n52 = ~\B[1]  & \B[2] ;
  assign new_n53 = ~\B[7]  & ~\B[8] ;
  assign new_n54 = new_n52 & new_n53;
  assign new_n55 = ~\B[9]  & ~new_n54;
  assign new_n56 = ~new_n51 & new_n55;
  assign new_n57 = ~\B[6]  & ~new_n56;
  assign new_n58 = \B[5]  & new_n57;
  assign new_n59 = \B[6]  & \B[9] ;
  assign new_n60 = ~\B[5]  & new_n59;
  assign new_n61 = ~new_n58 & ~new_n60;
  assign new_n62 = ~new_n45 & new_n61;
  assign new_n63 = ~\B[10]  & ~new_n62;
  assign new_n64 = ~\B[2]  & \B[3] ;
  assign new_n65 = ~\B[3]  & \B[2] ;
  assign new_n66 = ~new_n64 & ~new_n65;
  assign new_n67 = ~\B[9]  & ~new_n66;
  assign new_n68 = ~\B[8]  & new_n67;
  assign new_n69 = ~\B[10]  & ~new_n68;
  assign new_n70 = ~\B[7]  & ~new_n69;
  assign new_n71 = \B[9]  & \B[10] ;
  assign new_n72 = \B[8]  & new_n71;
  assign new_n73 = ~new_n70 & ~new_n72;
  assign new_n74 = ~new_n73 & \B[6] ;
  assign new_n75 = ~\B[6]  & \B[10] ;
  assign new_n76 = \B[7]  & new_n75;
  assign new_n77 = ~new_n74 & ~new_n76;
  assign \M[0]  = new_n63 | ~new_n77;
  assign new_n79 = ~\B[4]  & ~\B[9] ;
  assign new_n80 = ~\B[2]  & ~\B[7] ;
  assign new_n81 = ~new_n79 & ~new_n80;
  assign new_n82 = ~\B[1]  & ~new_n81;
  assign new_n83 = \B[1]  & \B[2] ;
  assign new_n84 = \B[0]  & new_n83;
  assign new_n85 = ~\B[0]  & ~\B[2] ;
  assign new_n86 = ~new_n84 & ~new_n85;
  assign new_n87 = ~\B[7]  & ~new_n86;
  assign new_n88 = \B[4]  & new_n87;
  assign new_n89 = ~\B[9]  & \B[8] ;
  assign new_n90 = ~new_n88 & ~new_n89;
  assign new_n91 = ~new_n82 & new_n90;
  assign new_n92 = ~\B[6]  & ~new_n91;
  assign new_n93 = \B[3]  & \B[4] ;
  assign new_n94 = ~new_n93 & \B[7] ;
  assign new_n95 = ~\B[9]  & new_n94;
  assign new_n96 = ~\B[8]  & new_n95;
  assign new_n97 = ~\B[7]  & \B[9] ;
  assign new_n98 = ~new_n96 & ~new_n97;
  assign new_n99 = ~new_n92 & new_n98;
  assign new_n100 = ~\B[5]  & ~new_n99;
  assign new_n101 = ~\B[8]  & ~\B[9] ;
  assign new_n102 = \B[4]  & new_n101;
  assign new_n103 = ~\B[6]  & ~\B[7] ;
  assign new_n104 = ~\B[4]  & new_n103;
  assign new_n105 = ~new_n102 & ~new_n104;
  assign new_n106 = ~new_n105 & \B[2] ;
  assign new_n107 = \B[1]  & new_n106;
  assign new_n108 = ~\B[9]  & \B[7] ;
  assign new_n109 = new_n46 & new_n108;
  assign new_n110 = ~new_n107 & ~new_n109;
  assign new_n111 = ~new_n110 & \B[3] ;
  assign new_n112 = \B[4]  & new_n89;
  assign new_n113 = \B[7]  & \B[9] ;
  assign new_n114 = ~new_n112 & ~new_n113;
  assign new_n115 = ~new_n114 & \B[6] ;
  assign new_n116 = ~new_n111 & ~new_n115;
  assign new_n117 = ~new_n116 & \B[5] ;
  assign new_n118 = ~\B[4]  & new_n89;
  assign new_n119 = ~new_n97 & ~new_n118;
  assign new_n120 = ~\B[6]  & ~new_n119;
  assign new_n121 = ~new_n117 & ~new_n120;
  assign new_n122 = ~new_n100 & new_n121;
  assign new_n123 = ~\B[10]  & ~new_n122;
  assign new_n124 = ~\B[9]  & \B[6] ;
  assign new_n125 = ~\B[4]  & new_n124;
  assign new_n126 = ~\B[6]  & \B[5] ;
  assign new_n127 = ~\B[3]  & new_n126;
  assign new_n128 = ~new_n125 & ~new_n127;
  assign new_n129 = ~\B[2]  & ~new_n128;
  assign new_n130 = ~\B[1]  & new_n126;
  assign new_n131 = ~new_n125 & ~new_n130;
  assign new_n132 = ~\B[3]  & ~new_n131;
  assign new_n133 = \B[2]  & \B[3] ;
  assign new_n134 = \B[4]  & new_n124;
  assign new_n135 = new_n133 & new_n134;
  assign new_n136 = ~\B[10]  & ~new_n135;
  assign new_n137 = ~new_n132 & new_n136;
  assign new_n138 = ~new_n129 & new_n137;
  assign new_n139 = ~\B[7]  & ~new_n138;
  assign new_n140 = ~new_n75 & ~new_n139;
  assign new_n141 = ~\B[8]  & ~new_n140;
  assign new_n142 = \B[6]  & \B[10] ;
  assign new_n143 = \B[7]  & new_n142;
  assign new_n144 = new_n89 & new_n143;
  assign new_n145 = ~new_n141 & ~new_n144;
  assign \M[1]  = ~new_n123 & new_n145;
  assign new_n147 = ~\B[6]  & \B[4] ;
  assign new_n148 = ~\B[3]  & \B[0] ;
  assign new_n149 = new_n147 & new_n148;
  assign new_n150 = ~\B[4]  & \B[5] ;
  assign new_n151 = \B[3]  & new_n150;
  assign new_n152 = ~new_n149 & ~new_n151;
  assign new_n153 = ~new_n152 & \B[1] ;
  assign new_n154 = ~\B[4]  & ~\B[6] ;
  assign new_n155 = \B[0]  & \B[1] ;
  assign new_n156 = ~new_n155 & \B[4] ;
  assign new_n157 = \B[3]  & new_n156;
  assign new_n158 = ~new_n154 & ~new_n157;
  assign new_n159 = ~\B[5]  & ~new_n158;
  assign new_n160 = ~new_n153 & ~new_n159;
  assign new_n161 = ~new_n160 & \B[2] ;
  assign new_n162 = ~\B[6]  & \B[3] ;
  assign new_n163 = ~\B[2]  & new_n162;
  assign new_n164 = ~\B[3]  & \B[5] ;
  assign new_n165 = ~new_n163 & ~new_n164;
  assign new_n166 = ~new_n165 & \B[4] ;
  assign new_n167 = ~new_n161 & ~new_n166;
  assign new_n168 = ~\B[7]  & ~new_n167;
  assign new_n169 = ~\B[5]  & \B[6] ;
  assign new_n170 = \B[2]  & new_n169;
  assign new_n171 = ~new_n130 & ~new_n170;
  assign new_n172 = ~new_n171 & \B[4] ;
  assign new_n173 = \B[3]  & new_n172;
  assign new_n174 = ~new_n93 & \B[5] ;
  assign new_n175 = \B[6]  & new_n174;
  assign new_n176 = ~new_n173 & ~new_n175;
  assign new_n177 = ~new_n168 & new_n176;
  assign new_n178 = ~\B[8]  & ~new_n177;
  assign new_n179 = ~\B[6]  & \B[7] ;
  assign new_n180 = \B[3]  & new_n179;
  assign new_n181 = ~\B[7]  & \B[6] ;
  assign new_n182 = ~\B[2]  & new_n181;
  assign new_n183 = ~new_n180 & ~new_n182;
  assign new_n184 = ~new_n183 & \B[5] ;
  assign new_n185 = \B[4]  & new_n184;
  assign new_n186 = \B[4]  & \B[5] ;
  assign new_n187 = ~new_n186 & \B[7] ;
  assign new_n188 = \B[6]  & new_n187;
  assign new_n189 = ~new_n185 & ~new_n188;
  assign new_n190 = ~new_n178 & new_n189;
  assign new_n191 = ~\B[9]  & ~new_n190;
  assign new_n192 = \B[4]  & \B[6] ;
  assign new_n193 = new_n33 & new_n192;
  assign new_n194 = ~new_n179 & ~new_n193;
  assign new_n195 = ~new_n194 & \B[8] ;
  assign new_n196 = ~new_n191 & ~new_n195;
  assign new_n197 = ~\B[10]  & ~new_n196;
  assign new_n198 = \B[8]  & \B[10] ;
  assign new_n199 = ~\B[8]  & \B[9] ;
  assign new_n200 = \B[5]  & new_n199;
  assign new_n201 = ~new_n198 & ~new_n200;
  assign new_n202 = ~new_n201 & \B[7] ;
  assign new_n203 = \B[6]  & new_n202;
  assign new_n204 = \B[5]  & \B[7] ;
  assign new_n205 = ~new_n204 & \B[8] ;
  assign new_n206 = ~\B[10]  & ~new_n205;
  assign new_n207 = ~new_n206 & \B[9] ;
  assign new_n208 = ~new_n203 & ~new_n207;
  assign \M[2]  = new_n197 | ~new_n208;
  assign new_n210 = \B[6]  & \B[7] ;
  assign new_n211 = ~\B[2]  & new_n210;
  assign new_n212 = \B[5]  & new_n28;
  assign new_n213 = new_n211 & new_n212;
  assign new_n214 = ~\B[5]  & new_n20;
  assign new_n215 = new_n103 & new_n214;
  assign new_n216 = ~new_n213 & ~new_n215;
  assign new_n217 = ~\B[9]  & ~new_n216;
  assign new_n218 = ~\B[10]  & new_n217;
  assign \M[3]  = \B[3]  | ~new_n218;
  assign new_n220 = \B[5]  & \B[6] ;
  assign new_n221 = ~\B[7]  & \B[4] ;
  assign new_n222 = new_n220 & new_n221;
  assign new_n223 = ~\B[5]  & ~\B[6] ;
  assign new_n224 = new_n155 & new_n223;
  assign new_n225 = ~new_n222 & ~new_n224;
  assign new_n226 = ~new_n225 & \B[3] ;
  assign new_n227 = \B[2]  & new_n226;
  assign new_n228 = ~\B[4]  & ~new_n181;
  assign new_n229 = ~\B[7]  & ~new_n126;
  assign new_n230 = ~\B[3]  & ~new_n229;
  assign new_n231 = ~new_n220 & \B[7] ;
  assign new_n232 = ~\B[6]  & ~new_n83;
  assign new_n233 = \B[5]  & new_n232;
  assign new_n234 = ~\B[9]  & ~new_n233;
  assign new_n235 = ~new_n231 & new_n234;
  assign new_n236 = ~new_n230 & new_n235;
  assign new_n237 = ~new_n228 & new_n236;
  assign new_n238 = ~new_n227 & new_n237;
  assign new_n239 = ~\B[8]  & ~new_n238;
  assign new_n240 = \B[3]  & \B[8] ;
  assign new_n241 = ~new_n65 & ~new_n240;
  assign new_n242 = ~new_n241 & \B[6] ;
  assign new_n243 = \B[5]  & new_n242;
  assign new_n244 = \B[7]  & new_n243;
  assign new_n245 = ~\B[9]  & new_n244;
  assign new_n246 = \B[4]  & new_n245;
  assign new_n247 = \B[7]  & new_n220;
  assign new_n248 = ~new_n247 & \B[9] ;
  assign new_n249 = ~new_n246 & ~new_n248;
  assign new_n250 = ~new_n239 & new_n249;
  assign \E[0]  = \B[10]  | new_n250;
  assign new_n252 = \B[6]  & \B[8] ;
  assign new_n253 = new_n204 & new_n252;
  assign new_n254 = \B[1]  & \B[3] ;
  assign new_n255 = \B[0]  & new_n254;
  assign new_n256 = ~\B[5]  & ~\B[7] ;
  assign new_n257 = ~\B[8]  & new_n256;
  assign new_n258 = new_n255 & new_n257;
  assign new_n259 = ~new_n253 & ~new_n258;
  assign new_n260 = ~new_n259 & \B[2] ;
  assign new_n261 = \B[8]  & new_n204;
  assign new_n262 = \B[3]  & \B[6] ;
  assign new_n263 = new_n261 & new_n262;
  assign new_n264 = ~new_n260 & ~new_n263;
  assign new_n265 = ~new_n264 & \B[4] ;
  assign new_n266 = new_n133 & new_n192;
  assign new_n267 = ~new_n266 & \B[5] ;
  assign new_n268 = ~new_n169 & ~new_n267;
  assign new_n269 = ~\B[7]  & ~new_n268;
  assign new_n270 = ~\B[8]  & new_n269;
  assign new_n271 = ~\B[9]  & ~\B[10] ;
  assign new_n272 = ~new_n270 & new_n271;
  assign \E[1]  = new_n265 | ~new_n272;
  assign new_n274 = \B[2]  & new_n93;
  assign new_n275 = new_n220 & new_n274;
  assign new_n276 = ~\B[9]  & ~new_n275;
  assign new_n277 = ~\B[10]  & new_n276;
  assign \E[2]  = ~new_n53 | ~new_n277;
endmodule


