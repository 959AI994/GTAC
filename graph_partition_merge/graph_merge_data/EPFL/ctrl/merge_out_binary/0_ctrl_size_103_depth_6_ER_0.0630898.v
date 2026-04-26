// Benchmark "ctrl" written by ABC on Thu Apr  2 14:52:09 2026

module ctrl ( 
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
    new_n41, new_n42, new_n44, new_n45, new_n46, new_n47, new_n49, new_n50,
    new_n51, new_n52, new_n54, new_n55, new_n56, new_n57, new_n58, new_n60,
    new_n61, new_n62, new_n63, new_n64, new_n65, new_n66, new_n67, new_n68,
    new_n69, new_n70, new_n71, new_n72, new_n74, new_n75, new_n77, new_n78,
    new_n79, new_n80, new_n81, new_n83, new_n84, new_n85, new_n87, new_n88,
    new_n89, new_n90, new_n92, new_n93, new_n95, new_n96, new_n97, new_n98,
    new_n99, new_n101, new_n102, new_n104, new_n105, new_n107, new_n110,
    new_n112, new_n114, new_n118, new_n119, new_n120, new_n121, new_n122,
    new_n124, new_n125, new_n126, new_n127, new_n129, new_n130, new_n132,
    new_n133, new_n135;
  assign new_n34 = ~\opcode[3]  & \opcode[4] ;
  assign new_n35 = ~new_n34 & \opcode[2] ;
  assign new_n36 = ~\opcode[3]  & \opcode[1] ;
  assign new_n37 = ~\opcode[4]  & ~new_n36;
  assign new_n38 = ~new_n37 & new_n35;
  assign new_n39 = \opcode[3]  & \opcode[4] ;
  assign new_n40 = ~\opcode[0]  & ~\opcode[1] ;
  assign new_n41 = ~\opcode[2]  & new_n39;
  assign new_n42 = ~new_n40 & new_n41;
  assign \sel_reg_dst[0]  = new_n38 | new_n42;
  assign new_n44 = ~new_n34 & \opcode[1] ;
  assign new_n45 = ~\opcode[0]  & new_n39;
  assign new_n46 = ~\opcode[1]  & ~new_n45;
  assign new_n47 = ~\opcode[2]  & ~new_n46;
  assign \sel_reg_dst[1]  = ~new_n44 & new_n47;
  assign new_n49 = ~\opcode[4]  & \opcode[3] ;
  assign new_n50 = ~\opcode[0]  & new_n34;
  assign new_n51 = ~new_n50 & \opcode[1] ;
  assign new_n52 = ~new_n49 & new_n51;
  assign \sel_alu_opB[0]  = ~new_n52 & new_n47;
  assign new_n54 = ~\opcode[3]  & ~\opcode[4] ;
  assign new_n55 = \opcode[0]  & \opcode[4] ;
  assign new_n56 = ~\opcode[1]  & ~\opcode[2] ;
  assign new_n57 = ~new_n55 & new_n56;
  assign new_n58 = ~new_n57 & \opcode[3] ;
  assign \sel_alu_opB[1]  = ~new_n54 & ~new_n58;
  assign new_n60 = \op_ext[0]  & new_n39;
  assign new_n61 = ~\opcode[0]  & new_n60;
  assign new_n62 = ~\op_ext[1]  & \opcode[4] ;
  assign new_n63 = ~new_n62 & \opcode[3] ;
  assign new_n64 = ~\op_ext[0]  & \opcode[4] ;
  assign new_n65 = \op_ext[1]  & new_n64;
  assign new_n66 = ~new_n65 & new_n63;
  assign new_n67 = \opcode[0]  & new_n66;
  assign new_n68 = ~\opcode[2]  & ~new_n61;
  assign new_n69 = ~new_n67 & new_n68;
  assign new_n70 = ~\opcode[0]  & \opcode[2] ;
  assign new_n71 = ~new_n56 & ~new_n70;
  assign new_n72 = ~new_n35 & new_n71;
  assign \alu_op[0]  = ~new_n69 & new_n72;
  assign new_n74 = ~\opcode[2]  & ~new_n63;
  assign new_n75 = ~new_n35 & \opcode[1] ;
  assign \alu_op[1]  = ~new_n74 & new_n75;
  assign new_n77 = \opcode[4]  & new_n35;
  assign new_n78 = \opcode[0]  & \opcode[1] ;
  assign new_n79 = ~new_n78 & new_n39;
  assign new_n80 = ~\opcode[2]  & ~new_n54;
  assign new_n81 = ~new_n79 & new_n80;
  assign \alu_op[2]  = new_n77 | new_n81;
  assign new_n83 = ~new_n36 & ~new_n55;
  assign new_n84 = ~new_n83 & new_n35;
  assign new_n85 = new_n45 & new_n56;
  assign \alu_op_ext[0]  = new_n84 | new_n85;
  assign new_n87 = ~new_n39 & \opcode[2] ;
  assign new_n88 = ~new_n54 & new_n87;
  assign new_n89 = ~\opcode[2]  & new_n51;
  assign new_n90 = ~new_n88 & \opcode[1] ;
  assign \alu_op_ext[1]  = ~new_n89 & new_n90;
  assign new_n92 = ~\opcode[1]  & ~new_n39;
  assign new_n93 = ~\opcode[2]  & ~new_n92;
  assign \alu_op_ext[2]  = ~new_n51 & new_n93;
  assign new_n95 = ~\opcode[3]  & ~new_n55;
  assign new_n96 = ~new_n95 & \opcode[1] ;
  assign new_n97 = ~new_n54 & new_n92;
  assign new_n98 = ~\opcode[2]  & ~new_n96;
  assign new_n99 = ~new_n97 & new_n98;
  assign \alu_op_ext[3]  = ~new_n35 & ~new_n99;
  assign new_n101 = ~\opcode[2]  & ~\opcode[3] ;
  assign new_n102 = new_n40 & new_n101;
  assign halt = ~\opcode[4]  & new_n102;
  assign new_n104 = ~\opcode[2]  & \opcode[3] ;
  assign new_n105 = ~\opcode[4]  & ~new_n104;
  assign reg_write = ~new_n102 & ~new_n105;
  assign new_n107 = \opcode[0]  & \opcode[2] ;
  assign sel_pc_opA = new_n54 & new_n107;
  assign sel_pc_opB = new_n54 & new_n70;
  assign new_n110 = ~\opcode[1]  & new_n49;
  assign beqz = new_n70 & new_n110;
  assign new_n112 = \opcode[0]  & new_n110;
  assign bnez = \opcode[2]  & new_n112;
  assign new_n114 = \opcode[1]  & new_n49;
  assign bgez = new_n107 & new_n114;
  assign bltz = new_n70 & new_n114;
  assign jump = ~\opcode[3]  & new_n35;
  assign new_n118 = ~new_n79 & \opcode[2] ;
  assign new_n119 = ~new_n64 & \opcode[3] ;
  assign new_n120 = new_n78 & new_n119;
  assign new_n121 = ~\opcode[2]  & ~new_n112;
  assign new_n122 = ~new_n120 & new_n121;
  assign Cin = ~new_n118 & ~new_n122;
  assign new_n124 = ~\op_ext[1]  & \opcode[1] ;
  assign new_n125 = new_n60 & new_n124;
  assign new_n126 = ~new_n110 & ~new_n125;
  assign new_n127 = ~\opcode[2]  & \opcode[0] ;
  assign invA = ~new_n126 & new_n127;
  assign new_n129 = new_n66 & new_n78;
  assign new_n130 = ~\opcode[2]  & ~new_n129;
  assign invB = ~new_n118 & ~new_n130;
  assign new_n132 = ~new_n40 & ~new_n78;
  assign new_n133 = ~\opcode[2]  & new_n34;
  assign mem_write = ~new_n132 & new_n133;
  assign new_n135 = ~\opcode[1]  & new_n34;
  assign sel_wb = new_n127 & new_n135;
  assign sign = 1'b1;
endmodule


