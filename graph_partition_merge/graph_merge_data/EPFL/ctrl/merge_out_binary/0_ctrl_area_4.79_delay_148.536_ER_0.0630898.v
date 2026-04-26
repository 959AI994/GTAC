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
    new_n41, new_n42, new_n43, new_n45, new_n46, new_n48, new_n49, new_n51,
    new_n52, new_n53, new_n54, new_n55, new_n57, new_n58, new_n59, new_n60,
    new_n61, new_n62, new_n63, new_n66, new_n67, new_n69, new_n71, new_n74,
    new_n75, new_n76, new_n78, new_n83, new_n84, new_n90, new_n91, new_n92,
    new_n93, new_n95;
  NOR2xp33_ASAP7_75t_R      g00(.A(\opcode[0] ), .B(\opcode[1] ), .Y(new_n34));
  INVx1_ASAP7_75t_R         g01(.A(\opcode[3] ), .Y(new_n35));
  INVx1_ASAP7_75t_R         g02(.A(\opcode[4] ), .Y(new_n36));
  NOR2xp33_ASAP7_75t_R      g03(.A(new_n35), .B(new_n36), .Y(new_n37));
  INVx1_ASAP7_75t_R         g04(.A(new_n37), .Y(new_n38));
  AOI21xp33_ASAP7_75t_R     g05(.A1(\opcode[1] ), .A2(new_n35), .B(\opcode[4] ), .Y(new_n39));
  INVx1_ASAP7_75t_R         g06(.A(\opcode[2] ), .Y(new_n40));
  NOR2xp33_ASAP7_75t_R      g07(.A(\opcode[3] ), .B(new_n36), .Y(new_n41));
  NOR2xp33_ASAP7_75t_R      g08(.A(new_n40), .B(new_n41), .Y(new_n42));
  INVx1_ASAP7_75t_R         g09(.A(new_n42), .Y(new_n43));
  OAI32xp33_ASAP7_75t_R     g10(.A1(\opcode[2] ), .A2(new_n34), .A3(new_n38), .B1(new_n39), .B2(new_n43), .Y(\sel_reg_dst[0] ));
  INVx1_ASAP7_75t_R         g11(.A(\opcode[1] ), .Y(new_n45));
  O2A1O1Ixp33_ASAP7_75t_R   g12(.A1(\opcode[0] ), .A2(new_n38), .B(new_n45), .C(\opcode[2] ), .Y(new_n46));
  OA21x2_ASAP7_75t_R        g13(.A1(new_n45), .A2(new_n41), .B(new_n46), .Y(\sel_reg_dst[1] ));
  OAI31xp33_ASAP7_75t_R     g14(.A1(\opcode[3] ), .A2(new_n36), .A3(\opcode[0] ), .B(\opcode[1] ), .Y(new_n48));
  A2O1A1Ixp33_ASAP7_75t_R   g15(.A1(\opcode[3] ), .A2(new_n36), .B(new_n48), .C(new_n46), .Y(new_n49));
  INVx1_ASAP7_75t_R         g16(.A(new_n49), .Y(\sel_alu_opB[0] ));
  INVx1_ASAP7_75t_R         g17(.A(\opcode[0] ), .Y(new_n51));
  NOR2xp33_ASAP7_75t_R      g18(.A(new_n51), .B(new_n36), .Y(new_n52));
  NOR2xp33_ASAP7_75t_R      g19(.A(\opcode[1] ), .B(\opcode[2] ), .Y(new_n53));
  INVx1_ASAP7_75t_R         g20(.A(new_n53), .Y(new_n54));
  NOR2xp33_ASAP7_75t_R      g21(.A(\opcode[3] ), .B(\opcode[4] ), .Y(new_n55));
  O2A1O1Ixp33_ASAP7_75t_R   g22(.A1(new_n52), .A2(new_n54), .B(\opcode[3] ), .C(new_n55), .Y(\sel_alu_opB[1] ));
  NOR2xp33_ASAP7_75t_R      g23(.A(\opcode[0] ), .B(new_n40), .Y(new_n57));
  INVx1_ASAP7_75t_R         g24(.A(\op_ext[0] ), .Y(new_n58));
  INVx1_ASAP7_75t_R         g25(.A(\op_ext[1] ), .Y(new_n59));
  AOI21xp33_ASAP7_75t_R     g26(.A1(\opcode[4] ), .A2(new_n59), .B(new_n35), .Y(new_n60));
  OAI31xp33_ASAP7_75t_R     g27(.A1(new_n36), .A2(\op_ext[0] ), .A3(new_n59), .B(new_n60), .Y(new_n61));
  OAI321xp33_ASAP7_75t_R    g28(.A1(new_n58), .A2(new_n38), .A3(\opcode[0] ), .B1(new_n51), .B2(new_n61), .C(new_n40), .Y(new_n62));
  INVx1_ASAP7_75t_R         g29(.A(new_n62), .Y(new_n63));
  NOR4xp25_ASAP7_75t_R      g30(.A(new_n53), .B(new_n57), .C(new_n42), .D(new_n63), .Y(\alu_op[0] ));
  OA211x2_ASAP7_75t_R       g31(.A1(\opcode[2] ), .A2(new_n60), .B(\opcode[1] ), .C(new_n43), .Y(\alu_op[1] ));
  NOR2xp33_ASAP7_75t_R      g32(.A(new_n51), .B(new_n45), .Y(new_n66));
  NOR2xp33_ASAP7_75t_R      g33(.A(new_n38), .B(new_n66), .Y(new_n67));
  OAI32xp33_ASAP7_75t_R     g34(.A1(\opcode[2] ), .A2(new_n55), .A3(new_n67), .B1(new_n36), .B2(new_n43), .Y(\alu_op[2] ));
  AOI21xp33_ASAP7_75t_R     g35(.A1(\opcode[1] ), .A2(new_n35), .B(new_n52), .Y(new_n69));
  OAI32xp33_ASAP7_75t_R     g36(.A1(\opcode[0] ), .A2(new_n38), .A3(new_n54), .B1(new_n43), .B2(new_n69), .Y(\alu_op_ext[0] ));
  OAI321xp33_ASAP7_75t_R    g37(.A1(new_n40), .A2(new_n37), .A3(new_n55), .B1(\opcode[2] ), .B2(new_n48), .C(\opcode[1] ), .Y(new_n71));
  INVx1_ASAP7_75t_R         g38(.A(new_n71), .Y(\alu_op_ext[1] ));
  OA211x2_ASAP7_75t_R       g39(.A1(\opcode[1] ), .A2(new_n37), .B(new_n40), .C(new_n48), .Y(\alu_op_ext[2] ));
  OAI21xp33_ASAP7_75t_R     g40(.A1(\opcode[3] ), .A2(new_n52), .B(\opcode[1] ), .Y(new_n74));
  INVx1_ASAP7_75t_R         g41(.A(new_n55), .Y(new_n75));
  AOI31xp33_ASAP7_75t_R     g42(.A1(new_n45), .A2(new_n38), .A3(new_n75), .B(\opcode[2] ), .Y(new_n76));
  AOI21xp33_ASAP7_75t_R     g43(.A1(new_n74), .A2(new_n76), .B(new_n42), .Y(\alu_op_ext[3] ));
  NOR4xp25_ASAP7_75t_R      g44(.A(\opcode[0] ), .B(\opcode[1] ), .C(\opcode[2] ), .D(\opcode[3] ), .Y(new_n78));
  AND2x2_ASAP7_75t_R        g45(.A(new_n36), .B(new_n78), .Y(halt));
  O2A1O1Ixp33_ASAP7_75t_R   g46(.A1(\opcode[2] ), .A2(new_n35), .B(new_n36), .C(new_n78), .Y(reg_write));
  NOR3xp33_ASAP7_75t_R      g47(.A(new_n51), .B(new_n40), .C(new_n75), .Y(sel_pc_opA));
  NOR3xp33_ASAP7_75t_R      g48(.A(\opcode[0] ), .B(new_n40), .C(new_n75), .Y(sel_pc_opB));
  NOR3xp33_ASAP7_75t_R      g49(.A(new_n35), .B(\opcode[4] ), .C(\opcode[1] ), .Y(new_n83));
  INVx1_ASAP7_75t_R         g50(.A(new_n83), .Y(new_n84));
  NOR3xp33_ASAP7_75t_R      g51(.A(\opcode[0] ), .B(new_n40), .C(new_n84), .Y(beqz));
  NOR3xp33_ASAP7_75t_R      g52(.A(new_n51), .B(new_n84), .C(new_n40), .Y(bnez));
  NOR5xp2_ASAP7_75t_R       g53(.A(new_n35), .B(\opcode[4] ), .C(new_n45), .D(new_n51), .E(new_n40), .Y(bgez));
  NOR5xp2_ASAP7_75t_R       g54(.A(new_n35), .B(\opcode[4] ), .C(new_n45), .D(\opcode[0] ), .E(new_n40), .Y(bltz));
  NOR2xp33_ASAP7_75t_R      g55(.A(\opcode[3] ), .B(new_n43), .Y(jump));
  INVx1_ASAP7_75t_R         g56(.A(new_n66), .Y(new_n90));
  OAI21xp33_ASAP7_75t_R     g57(.A1(new_n36), .A2(\op_ext[0] ), .B(\opcode[3] ), .Y(new_n91));
  AOI21xp33_ASAP7_75t_R     g58(.A1(\opcode[0] ), .A2(new_n83), .B(\opcode[2] ), .Y(new_n92));
  NOR2xp33_ASAP7_75t_R      g59(.A(new_n40), .B(new_n67), .Y(new_n93));
  O2A1O1Ixp33_ASAP7_75t_R   g60(.A1(new_n90), .A2(new_n91), .B(new_n92), .C(new_n93), .Y(Cin));
  NOR4xp25_ASAP7_75t_R      g61(.A(new_n58), .B(new_n38), .C(new_n45), .D(\op_ext[1] ), .Y(new_n95));
  OA211x2_ASAP7_75t_R       g62(.A1(new_n83), .A2(new_n95), .B(\opcode[0] ), .C(new_n40), .Y(invA));
  O2A1O1Ixp33_ASAP7_75t_R   g63(.A1(new_n61), .A2(new_n90), .B(new_n40), .C(new_n93), .Y(invB));
  assign                    sign = 1'b1;
  OA211x2_ASAP7_75t_R       g64(.A1(new_n34), .A2(new_n66), .B(new_n40), .C(new_n41), .Y(mem_write));
  NOR5xp2_ASAP7_75t_R       g65(.A(\opcode[3] ), .B(new_n36), .C(\opcode[1] ), .D(new_n51), .E(\opcode[2] ), .Y(sel_wb));
endmodule


