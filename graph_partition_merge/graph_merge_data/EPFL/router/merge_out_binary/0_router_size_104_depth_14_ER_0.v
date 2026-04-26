// Benchmark "router" written by ABC on Thu Apr  2 15:01:26 2026

module router ( 
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
    new_n140, new_n142, new_n143, new_n144, new_n145, new_n146, new_n147,
    new_n148, new_n149, new_n150, new_n151, new_n152, new_n153, new_n154,
    new_n155, new_n156, new_n157, new_n158, new_n159, new_n160, new_n161,
    new_n162, new_n163, new_n164, new_n165, new_n166, new_n167, new_n168,
    new_n169, new_n170, new_n171, new_n172, new_n173, new_n174, new_n175,
    new_n176, new_n177, new_n178, new_n179, new_n180, new_n181, new_n182,
    new_n183, new_n184, new_n185, new_n186, new_n187, new_n188, new_n189,
    new_n190, new_n191, new_n192;
  assign new_n91 = \dest_x[27]  & \dest_x[28] ;
  assign new_n92 = ~\dest_x[9]  & ~\dest_x[10] ;
  assign new_n93 = ~new_n92 & \dest_x[11] ;
  assign new_n94 = ~\dest_x[12]  & ~\dest_x[13] ;
  assign new_n95 = ~new_n93 & new_n94;
  assign new_n96 = \dest_x[14]  & \dest_x[15] ;
  assign new_n97 = ~new_n95 & new_n96;
  assign new_n98 = ~\dest_x[16]  & ~new_n97;
  assign new_n99 = ~new_n98 & \dest_x[17] ;
  assign new_n100 = ~\dest_x[18]  & ~new_n99;
  assign new_n101 = \dest_x[19]  & \dest_x[20] ;
  assign new_n102 = ~new_n100 & new_n101;
  assign new_n103 = ~\dest_x[21]  & ~\dest_x[22] ;
  assign new_n104 = ~new_n102 & new_n103;
  assign new_n105 = \dest_x[23]  & \dest_x[24] ;
  assign new_n106 = \dest_x[25]  & new_n105;
  assign new_n107 = ~new_n104 & new_n106;
  assign new_n108 = ~\dest_x[26]  & ~new_n107;
  assign new_n109 = \dest_x[29]  & new_n91;
  assign new_n110 = ~new_n108 & new_n109;
  assign new_n111 = ~\dest_x[23]  & ~\dest_x[24] ;
  assign new_n112 = ~\dest_x[25]  & new_n111;
  assign new_n113 = new_n104 & new_n112;
  assign new_n114 = ~new_n113 & \dest_x[26] ;
  assign new_n115 = \dest_x[11]  & \dest_x[12] ;
  assign new_n116 = \dest_x[16]  & new_n97;
  assign new_n117 = \dest_x[12]  & new_n92;
  assign new_n118 = ~new_n93 & ~new_n117;
  assign new_n119 = ~\dest_x[1]  & ~\dest_x[2] ;
  assign new_n120 = ~\dest_x[3]  & ~\dest_x[4] ;
  assign new_n121 = ~\dest_x[5]  & ~\dest_x[6] ;
  assign new_n122 = ~\dest_x[7]  & ~\dest_x[8] ;
  assign new_n123 = ~\dest_x[10]  & \dest_x[9] ;
  assign new_n124 = ~\dest_x[13]  & \dest_x[14] ;
  assign new_n125 = \dest_x[15]  & \dest_x[17] ;
  assign new_n126 = ~\dest_x[18]  & new_n125;
  assign new_n127 = new_n123 & new_n124;
  assign new_n128 = new_n121 & new_n122;
  assign new_n129 = new_n119 & new_n120;
  assign new_n130 = new_n91 & new_n101;
  assign new_n131 = ~new_n115 & new_n103;
  assign new_n132 = new_n130 & new_n131;
  assign new_n133 = new_n128 & new_n129;
  assign new_n134 = new_n126 & new_n127;
  assign new_n135 = new_n133 & new_n134;
  assign new_n136 = ~new_n118 & new_n132;
  assign new_n137 = new_n135 & new_n136;
  assign new_n138 = ~new_n98 & new_n137;
  assign new_n139 = ~new_n116 & new_n138;
  assign new_n140 = ~new_n114 & new_n139;
  assign \outport[0]  = ~new_n110 | ~new_n140;
  assign new_n142 = \dest_y[14]  & \dest_y[15] ;
  assign new_n143 = ~\dest_y[9]  & ~\dest_y[10] ;
  assign new_n144 = ~new_n143 & \dest_y[11] ;
  assign new_n145 = ~\dest_y[12]  & ~\dest_y[13] ;
  assign new_n146 = ~new_n144 & new_n145;
  assign new_n147 = ~new_n146 & new_n142;
  assign new_n148 = ~\dest_y[16]  & ~new_n147;
  assign new_n149 = ~new_n148 & \dest_y[17] ;
  assign new_n150 = ~\dest_y[18]  & ~new_n149;
  assign new_n151 = ~\dest_y[19]  & ~new_n150;
  assign new_n152 = \dest_y[19]  & new_n150;
  assign new_n153 = ~new_n151 & ~new_n152;
  assign new_n154 = ~\dest_y[11]  & new_n143;
  assign new_n155 = \dest_y[26]  & \dest_y[27] ;
  assign new_n156 = ~\dest_y[28]  & ~new_n155;
  assign new_n157 = ~new_n146 & \dest_y[17] ;
  assign new_n158 = ~\dest_y[17]  & new_n146;
  assign new_n159 = ~\dest_x[0]  & \dest_y[0] ;
  assign new_n160 = \dest_y[1]  & \dest_y[2] ;
  assign new_n161 = \dest_y[3]  & \dest_y[4] ;
  assign new_n162 = \dest_y[5]  & \dest_y[6] ;
  assign new_n163 = \dest_y[7]  & \dest_y[8] ;
  assign new_n164 = ~\dest_y[16]  & ~\dest_y[18] ;
  assign new_n165 = ~\dest_y[21]  & \dest_y[20] ;
  assign new_n166 = ~\dest_y[22]  & \dest_y[23] ;
  assign new_n167 = \dest_y[24]  & \dest_y[25] ;
  assign new_n168 = ~\dest_y[26]  & \dest_y[27] ;
  assign new_n169 = \dest_y[29]  & new_n168;
  assign new_n170 = new_n166 & new_n167;
  assign new_n171 = new_n164 & new_n165;
  assign new_n172 = new_n145 & new_n163;
  assign new_n173 = new_n161 & new_n162;
  assign new_n174 = new_n159 & new_n160;
  assign new_n175 = new_n142 & new_n143;
  assign new_n176 = new_n174 & new_n175;
  assign new_n177 = new_n172 & new_n173;
  assign new_n178 = new_n170 & new_n171;
  assign new_n179 = ~new_n144 & new_n169;
  assign new_n180 = ~new_n154 & ~new_n156;
  assign new_n181 = new_n179 & new_n180;
  assign new_n182 = new_n177 & new_n178;
  assign new_n183 = new_n176 & new_n182;
  assign new_n184 = ~new_n157 & new_n181;
  assign new_n185 = ~new_n158 & new_n184;
  assign new_n186 = new_n183 & new_n185;
  assign new_n187 = ~new_n153 & new_n186;
  assign new_n188 = \dest_y[28]  & \dest_y[29] ;
  assign new_n189 = new_n155 & new_n188;
  assign new_n190 = ~new_n187 & ~new_n189;
  assign new_n191 = ~new_n108 & new_n190;
  assign new_n192 = new_n140 & new_n191;
  assign \outport[1]  = new_n192 | ~new_n110;
  assign \outport[2]  = ~\outport[0]  & new_n189;
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


