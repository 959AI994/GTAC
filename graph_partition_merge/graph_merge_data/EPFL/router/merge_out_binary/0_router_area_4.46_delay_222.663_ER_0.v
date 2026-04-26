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
    new_n120, new_n121, new_n122, new_n123, new_n124, new_n125, new_n126,
    new_n127, new_n128, new_n129, new_n130, new_n131, new_n132, new_n133,
    new_n134, new_n135, new_n136, new_n137, new_n138, new_n139, new_n140,
    new_n141, new_n142, new_n143, new_n144, new_n145, new_n146, new_n147,
    new_n149;
  INVx1_ASAP7_75t_R         g00(.A(\dest_x[16] ), .Y(new_n91));
  OA21x2_ASAP7_75t_R        g01(.A1(\dest_x[9] ), .A2(\dest_x[10] ), .B(\dest_x[11] ), .Y(new_n92));
  OAI311xp33_ASAP7_75t_R    g02(.A1(\dest_x[12] ), .A2(\dest_x[13] ), .A3(new_n92), .B1(\dest_x[14] ), .C1(\dest_x[15] ), .Y(new_n93));
  NAND2xp33_ASAP7_75t_R     g03(.A(new_n91), .B(new_n93), .Y(new_n94));
  AO21x1_ASAP7_75t_R        g04(.A1(\dest_x[17] ), .A2(new_n94), .B(\dest_x[18] ), .Y(new_n95));
  AOI311xp33_ASAP7_75t_R    g05(.A1(\dest_x[19] ), .A2(\dest_x[20] ), .A3(new_n95), .B(\dest_x[21] ), .C(\dest_x[22] ), .Y(new_n96));
  NAND3xp33_ASAP7_75t_R     g06(.A(\dest_x[23] ), .B(\dest_x[24] ), .C(\dest_x[25] ), .Y(new_n97));
  INVx1_ASAP7_75t_R         g07(.A(\dest_x[26] ), .Y(new_n98));
  OAI21xp33_ASAP7_75t_R     g08(.A1(new_n96), .A2(new_n97), .B(new_n98), .Y(new_n99));
  NAND4xp25_ASAP7_75t_R     g09(.A(\dest_x[27] ), .B(\dest_x[28] ), .C(\dest_x[29] ), .D(new_n99), .Y(new_n100));
  INVx1_ASAP7_75t_R         g10(.A(new_n96), .Y(new_n101));
  NOR4xp25_ASAP7_75t_R      g11(.A(\dest_x[23] ), .B(\dest_x[24] ), .C(\dest_x[25] ), .D(new_n101), .Y(new_n102));
  OR4x1_ASAP7_75t_R         g12(.A(\dest_x[1] ), .B(\dest_x[2] ), .C(\dest_x[3] ), .D(\dest_x[4] ), .Y(new_n103));
  OR5x1_ASAP7_75t_R         g13(.A(\dest_x[5] ), .B(\dest_x[6] ), .C(\dest_x[7] ), .D(\dest_x[8] ), .E(new_n103), .Y(new_n104));
  INVx1_ASAP7_75t_R         g14(.A(\dest_x[18] ), .Y(new_n105));
  INVx1_ASAP7_75t_R         g15(.A(\dest_x[9] ), .Y(new_n106));
  INVx1_ASAP7_75t_R         g16(.A(\dest_x[14] ), .Y(new_n107));
  NOR4xp25_ASAP7_75t_R      g17(.A(new_n106), .B(\dest_x[10] ), .C(\dest_x[13] ), .D(new_n107), .Y(new_n108));
  NAND4xp25_ASAP7_75t_R     g18(.A(\dest_x[15] ), .B(\dest_x[17] ), .C(new_n105), .D(new_n108), .Y(new_n109));
  INVx1_ASAP7_75t_R         g19(.A(\dest_x[10] ), .Y(new_n110));
  AOI31xp33_ASAP7_75t_R     g20(.A1(new_n106), .A2(new_n110), .A3(\dest_x[12] ), .B(new_n92), .Y(new_n111));
  NOR2xp33_ASAP7_75t_R      g21(.A(\dest_x[21] ), .B(\dest_x[22] ), .Y(new_n112));
  NAND2xp33_ASAP7_75t_R     g22(.A(\dest_x[27] ), .B(\dest_x[28] ), .Y(new_n113));
  AOI21xp33_ASAP7_75t_R     g23(.A1(\dest_x[11] ), .A2(\dest_x[12] ), .B(new_n113), .Y(new_n114));
  NAND4xp25_ASAP7_75t_R     g24(.A(\dest_x[19] ), .B(\dest_x[20] ), .C(new_n112), .D(new_n114), .Y(new_n115));
  INVx1_ASAP7_75t_R         g25(.A(new_n94), .Y(new_n116));
  NOR5xp2_ASAP7_75t_R       g26(.A(new_n104), .B(new_n109), .C(new_n111), .D(new_n115), .E(new_n116), .Y(new_n117));
  OAI221xp5_ASAP7_75t_R     g27(.A1(new_n91), .A2(new_n93), .B1(new_n98), .B2(new_n102), .C(new_n117), .Y(new_n118));
  OR2x2_ASAP7_75t_R         g28(.A(new_n100), .B(new_n118), .Y(\outport[0] ));
  AND2x2_ASAP7_75t_R        g29(.A(\dest_y[26] ), .B(\dest_y[27] ), .Y(new_n120));
  AND4x1_ASAP7_75t_R        g30(.A(\dest_y[1] ), .B(\dest_y[2] ), .C(\dest_y[3] ), .D(\dest_y[4] ), .Y(new_n121));
  NAND5xp2_ASAP7_75t_R      g31(.A(\dest_y[5] ), .B(\dest_y[6] ), .C(\dest_y[7] ), .D(\dest_y[8] ), .E(new_n121), .Y(new_n122));
  INVx1_ASAP7_75t_R         g32(.A(\dest_y[22] ), .Y(new_n123));
  INVx1_ASAP7_75t_R         g33(.A(\dest_y[20] ), .Y(new_n124));
  NOR4xp25_ASAP7_75t_R      g34(.A(\dest_y[16] ), .B(\dest_y[18] ), .C(new_n124), .D(\dest_y[21] ), .Y(new_n125));
  NAND5xp2_ASAP7_75t_R      g35(.A(new_n123), .B(\dest_y[23] ), .C(\dest_y[24] ), .D(\dest_y[25] ), .E(new_n125), .Y(new_n126));
  INVx1_ASAP7_75t_R         g36(.A(\dest_y[0] ), .Y(new_n127));
  OR2x2_ASAP7_75t_R         g37(.A(\dest_y[9] ), .B(\dest_y[10] ), .Y(new_n128));
  OR2x2_ASAP7_75t_R         g38(.A(\dest_y[12] ), .B(\dest_y[13] ), .Y(new_n129));
  NAND2xp33_ASAP7_75t_R     g39(.A(\dest_y[14] ), .B(\dest_y[15] ), .Y(new_n130));
  OR5x1_ASAP7_75t_R         g40(.A(\dest_x[0] ), .B(new_n127), .C(new_n128), .D(new_n129), .E(new_n130), .Y(new_n131));
  AND2x2_ASAP7_75t_R        g41(.A(\dest_y[11] ), .B(new_n128), .Y(new_n132));
  INVx1_ASAP7_75t_R         g42(.A(\dest_y[17] ), .Y(new_n133));
  NOR2xp33_ASAP7_75t_R      g43(.A(new_n132), .B(new_n129), .Y(new_n134));
  INVx1_ASAP7_75t_R         g44(.A(\dest_y[27] ), .Y(new_n135));
  INVx1_ASAP7_75t_R         g45(.A(\dest_y[29] ), .Y(new_n136));
  OAI22xp33_ASAP7_75t_R     g46(.A1(\dest_y[11] ), .A2(new_n128), .B1(\dest_y[28] ), .B2(new_n120), .Y(new_n137));
  NOR5xp2_ASAP7_75t_R       g47(.A(\dest_y[26] ), .B(new_n135), .C(new_n136), .D(new_n132), .E(new_n137), .Y(new_n138));
  OAI321xp33_ASAP7_75t_R    g48(.A1(new_n132), .A2(new_n129), .A3(\dest_y[17] ), .B1(new_n133), .B2(new_n134), .C(new_n138), .Y(new_n139));
  NOR2xp33_ASAP7_75t_R      g49(.A(new_n134), .B(new_n130), .Y(new_n140));
  O2A1O1Ixp33_ASAP7_75t_R   g50(.A1(\dest_y[16] ), .A2(new_n140), .B(\dest_y[17] ), .C(\dest_y[18] ), .Y(new_n141));
  NOR2xp33_ASAP7_75t_R      g51(.A(\dest_y[19] ), .B(new_n141), .Y(new_n142));
  AOI21xp33_ASAP7_75t_R     g52(.A1(\dest_y[19] ), .A2(new_n141), .B(new_n142), .Y(new_n143));
  NOR5xp2_ASAP7_75t_R       g53(.A(new_n122), .B(new_n126), .C(new_n131), .D(new_n139), .E(new_n143), .Y(new_n144));
  AOI31xp33_ASAP7_75t_R     g54(.A1(\dest_y[28] ), .A2(\dest_y[29] ), .A3(new_n120), .B(new_n144), .Y(new_n145));
  INVx1_ASAP7_75t_R         g55(.A(new_n118), .Y(new_n146));
  AOI31xp33_ASAP7_75t_R     g56(.A1(new_n99), .A2(new_n145), .A3(new_n146), .B(new_n100), .Y(new_n147));
  INVx1_ASAP7_75t_R         g57(.A(new_n147), .Y(\outport[1] ));
  NAND3xp33_ASAP7_75t_R     g58(.A(\dest_y[28] ), .B(\dest_y[29] ), .C(new_n120), .Y(new_n149));
  NOR2xp33_ASAP7_75t_R      g59(.A(\outport[0] ), .B(new_n149), .Y(\outport[2] ));
  assign                    \outport[3]  = 1'b0;
  assign                    \outport[4]  = 1'b0;
  assign                    \outport[5]  = 1'b0;
  assign                    \outport[6]  = 1'b0;
  assign                    \outport[7]  = 1'b0;
  assign                    \outport[8]  = 1'b0;
  assign                    \outport[9]  = 1'b0;
  assign                    \outport[10]  = 1'b0;
  assign                    \outport[11]  = 1'b0;
  assign                    \outport[12]  = 1'b0;
  assign                    \outport[13]  = 1'b0;
  assign                    \outport[14]  = 1'b0;
  assign                    \outport[15]  = 1'b0;
  assign                    \outport[16]  = 1'b0;
  assign                    \outport[17]  = 1'b0;
  assign                    \outport[18]  = 1'b0;
  assign                    \outport[19]  = 1'b0;
  assign                    \outport[20]  = 1'b0;
  assign                    \outport[21]  = 1'b0;
  assign                    \outport[22]  = 1'b0;
  assign                    \outport[23]  = 1'b0;
  assign                    \outport[24]  = 1'b0;
  assign                    \outport[25]  = 1'b0;
  assign                    \outport[26]  = 1'b0;
  assign                    \outport[27]  = 1'b0;
  assign                    \outport[28]  = 1'b0;
  assign                    \outport[29]  = 1'b0;
endmodule


