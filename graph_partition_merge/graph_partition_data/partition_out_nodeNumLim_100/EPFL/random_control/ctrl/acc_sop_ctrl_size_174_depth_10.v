// Benchmark "../EPFL/benchmarks/random_control/ctrl" written by ABC on Wed Oct 29 18:32:53 2025

module \../EPFL/benchmarks/random_control/ctrl  ( 
    \opcode[0] , \opcode[1] , \opcode[2] , \opcode[3] , \opcode[4] ,
    \op_ext[0] , \op_ext[1] ,
    \sel_reg_dst[0] , \sel_reg_dst[1] , \sel_alu_opB[0] , \sel_alu_opB[1] ,
    \alu_op[0] , \alu_op[1] , \alu_op[2] , \alu_op_ext[0] ,
    \alu_op_ext[1] , \alu_op_ext[2] , \alu_op_ext[3] , halt, reg_write,
    sel_pc_opA, sel_pc_opB, beqz, bnez, bgez, bltz, jump, Cin, invA, invB,
    sign, mem_write, sel_wb  );
  input  \opcode[0] , \opcode[1] , \opcode[2] , \opcode[3] , \opcode[4] ,
    \op_ext[0] , \op_ext[1] ;
  output \sel_reg_dst[0] , \sel_reg_dst[1] , \sel_alu_opB[0] ,
    \sel_alu_opB[1] , \alu_op[0] , \alu_op[1] , \alu_op[2] ,
    \alu_op_ext[0] , \alu_op_ext[1] , \alu_op_ext[2] , \alu_op_ext[3] ,
    halt, reg_write, sel_pc_opA, sel_pc_opB, beqz, bnez, bgez, bltz, jump,
    Cin, invA, invB, sign, mem_write, sel_wb;
  wire new_n34, new_n35, new_n36, new_n37, new_n38, new_n39, new_n40,
    new_n41, new_n42, new_n43, new_n44, new_n45, new_n46, new_n47, new_n49,
    new_n50, new_n51, new_n52, new_n53, new_n54, new_n55, new_n56, new_n57,
    new_n58, new_n59, new_n60, new_n62, new_n63, new_n64, new_n65, new_n66,
    new_n67, new_n68, new_n70, new_n71, new_n72, new_n73, new_n74, new_n75,
    new_n76, new_n77, new_n79, new_n80, new_n81, new_n82, new_n83, new_n84,
    new_n85, new_n86, new_n87, new_n88, new_n89, new_n90, new_n91, new_n92,
    new_n93, new_n94, new_n95, new_n96, new_n98, new_n99, new_n100,
    new_n101, new_n102, new_n103, new_n105, new_n106, new_n107, new_n108,
    new_n109, new_n110, new_n111, new_n112, new_n113, new_n114, new_n116,
    new_n117, new_n118, new_n119, new_n120, new_n122, new_n123, new_n124,
    new_n125, new_n126, new_n127, new_n128, new_n130, new_n131, new_n133,
    new_n134, new_n135, new_n136, new_n138, new_n139, new_n140, new_n141,
    new_n142, new_n144, new_n145, new_n146, new_n147, new_n148, new_n149,
    new_n150, new_n152, new_n153, new_n155, new_n157, new_n158, new_n159,
    new_n160, new_n161, new_n163, new_n164, new_n165, new_n166, new_n168,
    new_n169, new_n170, new_n172, new_n173, new_n174, new_n176, new_n178,
    new_n179, new_n180, new_n181, new_n182, new_n183, new_n184, new_n185,
    new_n187, new_n188, new_n189, new_n190, new_n191, new_n192, new_n193,
    new_n194, new_n196, new_n197, new_n200, new_n201, new_n202, new_n203,
    new_n205, new_n206, new_n207;
  assign new_n34 = ~\opcode[1]  & \opcode[0] ;
  assign new_n35 = \opcode[3]  & \opcode[4] ;
  assign new_n36 = new_n34 & new_n35;
  assign new_n37 = \opcode[1]  & \opcode[3] ;
  assign new_n38 = \opcode[4]  & new_n37;
  assign new_n39 = ~new_n36 & ~new_n38;
  assign new_n40 = ~\opcode[2]  & ~new_n39;
  assign new_n41 = ~\opcode[1]  & \opcode[3] ;
  assign new_n42 = \opcode[4]  & new_n41;
  assign new_n43 = ~\opcode[3]  & ~\opcode[4] ;
  assign new_n44 = ~new_n35 & ~new_n43;
  assign new_n45 = ~new_n44 & \opcode[1] ;
  assign new_n46 = ~new_n42 & ~new_n45;
  assign new_n47 = ~new_n46 & \opcode[2] ;
  assign \sel_reg_dst[0]  = new_n40 | new_n47;
  assign new_n49 = ~\opcode[0]  & ~new_n35;
  assign new_n50 = ~\opcode[0]  & ~new_n49;
  assign new_n51 = ~\opcode[1]  & ~new_n50;
  assign new_n52 = ~\opcode[3]  & ~new_n43;
  assign new_n53 = ~new_n52 & \opcode[1] ;
  assign new_n54 = ~new_n51 & ~new_n53;
  assign new_n55 = ~\opcode[2]  & ~new_n54;
  assign new_n56 = ~\opcode[3]  & \opcode[4] ;
  assign new_n57 = ~\opcode[3]  & ~new_n56;
  assign new_n58 = ~new_n57 & \opcode[1] ;
  assign new_n59 = ~new_n58 & \opcode[1] ;
  assign new_n60 = ~new_n59 & \opcode[2] ;
  assign \sel_reg_dst[1]  = ~new_n55 & ~new_n60;
  assign new_n62 = ~\opcode[0]  & ~new_n44;
  assign new_n63 = ~new_n35 & \opcode[3] ;
  assign new_n64 = ~new_n63 & \opcode[0] ;
  assign new_n65 = ~new_n62 & ~new_n64;
  assign new_n66 = ~new_n65 & \opcode[1] ;
  assign new_n67 = ~new_n51 & ~new_n66;
  assign new_n68 = ~\opcode[2]  & ~new_n67;
  assign \sel_alu_opB[0]  = ~\opcode[2]  & ~new_n68;
  assign new_n70 = ~\opcode[0]  & ~\opcode[3] ;
  assign new_n71 = ~new_n56 & new_n70;
  assign new_n72 = ~new_n44 & \opcode[0] ;
  assign new_n73 = ~new_n71 & ~new_n72;
  assign new_n74 = ~\opcode[1]  & ~new_n73;
  assign new_n75 = ~new_n53 & ~new_n74;
  assign new_n76 = ~\opcode[2]  & ~new_n75;
  assign new_n77 = ~new_n52 & \opcode[2] ;
  assign \sel_alu_opB[1]  = ~new_n76 & ~new_n77;
  assign new_n79 = ~\opcode[0]  & \opcode[3] ;
  assign new_n80 = \opcode[4]  & \op_ext[0] ;
  assign new_n81 = new_n79 & new_n80;
  assign new_n82 = ~\op_ext[1]  & \opcode[3] ;
  assign new_n83 = ~new_n35 & new_n82;
  assign new_n84 = ~\op_ext[0]  & \opcode[3] ;
  assign new_n85 = ~new_n35 & new_n84;
  assign new_n86 = \opcode[3]  & \op_ext[0] ;
  assign new_n87 = ~new_n85 & ~new_n86;
  assign new_n88 = ~new_n87 & \op_ext[1] ;
  assign new_n89 = ~new_n83 & ~new_n88;
  assign new_n90 = ~new_n89 & \opcode[0] ;
  assign new_n91 = ~new_n81 & ~new_n90;
  assign new_n92 = ~new_n91 & \opcode[1] ;
  assign new_n93 = ~\opcode[2]  & ~new_n92;
  assign new_n94 = ~new_n52 & \opcode[0] ;
  assign new_n95 = ~new_n94 & \opcode[0] ;
  assign new_n96 = ~new_n95 & \opcode[2] ;
  assign \alu_op[0]  = ~new_n93 & ~new_n96;
  assign new_n98 = \opcode[3]  & \op_ext[1] ;
  assign new_n99 = ~new_n83 & ~new_n98;
  assign new_n100 = ~new_n99 & \opcode[1] ;
  assign new_n101 = ~\opcode[2]  & ~new_n100;
  assign new_n102 = ~new_n53 & \opcode[1] ;
  assign new_n103 = ~new_n102 & \opcode[2] ;
  assign \alu_op[1]  = ~new_n101 & ~new_n103;
  assign new_n105 = ~\opcode[1]  & ~new_n35;
  assign new_n106 = ~new_n43 & new_n105;
  assign new_n107 = ~new_n43 & new_n49;
  assign new_n108 = ~new_n57 & \opcode[0] ;
  assign new_n109 = ~new_n107 & ~new_n108;
  assign new_n110 = ~new_n109 & \opcode[1] ;
  assign new_n111 = ~new_n106 & ~new_n110;
  assign new_n112 = ~\opcode[2]  & ~new_n111;
  assign new_n113 = \opcode[2]  & \opcode[3] ;
  assign new_n114 = \opcode[4]  & new_n113;
  assign \alu_op[2]  = new_n112 | new_n114;
  assign new_n116 = ~\opcode[1]  & ~\opcode[2] ;
  assign new_n117 = ~new_n51 & new_n116;
  assign new_n118 = ~new_n73 & \opcode[1] ;
  assign new_n119 = ~new_n36 & ~new_n118;
  assign new_n120 = ~new_n119 & \opcode[2] ;
  assign \alu_op_ext[0]  = new_n117 | new_n120;
  assign new_n122 = ~\opcode[0]  & ~new_n52;
  assign new_n123 = ~\opcode[0]  & ~new_n122;
  assign new_n124 = ~new_n123 & \opcode[1] ;
  assign new_n125 = ~\opcode[2]  & \opcode[1] ;
  assign new_n126 = ~new_n124 & new_n125;
  assign new_n127 = \opcode[1]  & \opcode[2] ;
  assign new_n128 = ~new_n44 & new_n127;
  assign \alu_op_ext[1]  = new_n126 | new_n128;
  assign new_n130 = ~new_n105 & ~new_n124;
  assign new_n131 = ~\opcode[2]  & ~new_n130;
  assign \alu_op_ext[2]  = ~new_n60 & ~new_n131;
  assign new_n133 = ~new_n79 & ~new_n108;
  assign new_n134 = ~new_n133 & \opcode[1] ;
  assign new_n135 = ~\opcode[2]  & ~new_n106;
  assign new_n136 = ~new_n134 & new_n135;
  assign \alu_op_ext[3]  = ~new_n77 & ~new_n136;
  assign new_n138 = ~\opcode[0]  & ~new_n57;
  assign new_n139 = ~\opcode[0]  & ~new_n138;
  assign new_n140 = ~\opcode[1]  & ~new_n139;
  assign new_n141 = ~\opcode[1]  & ~new_n140;
  assign new_n142 = ~\opcode[2]  & ~new_n141;
  assign halt = ~\opcode[2]  & ~new_n142;
  assign new_n144 = ~\opcode[1]  & ~new_n133;
  assign new_n145 = ~new_n58 & ~new_n144;
  assign new_n146 = ~\opcode[2]  & ~new_n145;
  assign new_n147 = ~\opcode[1]  & \opcode[4] ;
  assign new_n148 = ~new_n63 & \opcode[1] ;
  assign new_n149 = ~new_n147 & ~new_n148;
  assign new_n150 = ~new_n149 & \opcode[2] ;
  assign reg_write = new_n146 | new_n150;
  assign new_n152 = ~new_n108 & \opcode[0] ;
  assign new_n153 = ~new_n152 & \opcode[2] ;
  assign sel_pc_opA = ~new_n153 & \opcode[2] ;
  assign new_n155 = ~new_n139 & \opcode[2] ;
  assign sel_pc_opB = ~new_n155 & \opcode[2] ;
  assign new_n157 = ~\opcode[0]  & ~new_n63;
  assign new_n158 = ~\opcode[0]  & ~new_n157;
  assign new_n159 = ~\opcode[1]  & ~new_n158;
  assign new_n160 = ~\opcode[1]  & ~new_n159;
  assign new_n161 = ~new_n160 & \opcode[2] ;
  assign beqz = ~new_n161 & \opcode[2] ;
  assign new_n163 = ~new_n64 & \opcode[0] ;
  assign new_n164 = ~\opcode[1]  & ~new_n163;
  assign new_n165 = ~\opcode[1]  & ~new_n164;
  assign new_n166 = ~new_n165 & \opcode[2] ;
  assign bnez = ~new_n166 & \opcode[2] ;
  assign new_n168 = ~new_n163 & \opcode[1] ;
  assign new_n169 = ~new_n168 & \opcode[1] ;
  assign new_n170 = ~new_n169 & \opcode[2] ;
  assign bgez = ~new_n170 & \opcode[2] ;
  assign new_n172 = ~new_n158 & \opcode[1] ;
  assign new_n173 = ~new_n172 & \opcode[1] ;
  assign new_n174 = ~new_n173 & \opcode[2] ;
  assign bltz = ~new_n174 & \opcode[2] ;
  assign new_n176 = ~new_n57 & \opcode[2] ;
  assign jump = ~new_n176 & \opcode[2] ;
  assign new_n178 = \opcode[0]  & \opcode[1] ;
  assign new_n179 = ~new_n87 & new_n178;
  assign new_n180 = ~new_n64 & new_n34;
  assign new_n181 = ~\opcode[2]  & ~new_n180;
  assign new_n182 = ~new_n179 & new_n181;
  assign new_n183 = ~new_n50 & \opcode[1] ;
  assign new_n184 = ~new_n105 & ~new_n183;
  assign new_n185 = ~new_n184 & \opcode[2] ;
  assign Cin = ~new_n182 & ~new_n185;
  assign new_n187 = \op_ext[0]  & new_n35;
  assign new_n188 = ~\op_ext[1]  & ~new_n187;
  assign new_n189 = ~\op_ext[1]  & ~new_n188;
  assign new_n190 = ~new_n189 & \opcode[0] ;
  assign new_n191 = ~new_n190 & \opcode[0] ;
  assign new_n192 = ~new_n191 & \opcode[1] ;
  assign new_n193 = ~new_n164 & ~new_n192;
  assign new_n194 = ~\opcode[2]  & ~new_n193;
  assign invA = ~\opcode[2]  & ~new_n194;
  assign new_n196 = ~new_n89 & new_n178;
  assign new_n197 = ~\opcode[2]  & ~new_n196;
  assign invB = ~new_n185 & ~new_n197;
  assign sign = 1'b1;
  assign new_n200 = ~\opcode[1]  & ~new_n123;
  assign new_n201 = ~new_n95 & \opcode[1] ;
  assign new_n202 = ~new_n200 & ~new_n201;
  assign new_n203 = ~\opcode[2]  & ~new_n202;
  assign mem_write = ~\opcode[2]  & ~new_n203;
  assign new_n205 = ~\opcode[1]  & ~new_n95;
  assign new_n206 = ~\opcode[1]  & ~new_n205;
  assign new_n207 = ~\opcode[2]  & ~new_n206;
  assign sel_wb = ~\opcode[2]  & ~new_n207;
endmodule


