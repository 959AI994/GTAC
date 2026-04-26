// Benchmark "router" written by ABC on Thu Apr  2 15:01:27 2026

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
    new_n148, new_n149;
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
  INVx1_ASAP7_75t_R         g29(.A(\dest_y[17] ), .Y(new_n120));
  NOR2xp33_ASAP7_75t_R      g30(.A(\dest_y[9] ), .B(\dest_y[10] ), .Y(new_n121));
  INVx1_ASAP7_75t_R         g31(.A(new_n121), .Y(new_n122));
  AO21x1_ASAP7_75t_R        g32(.A1(\dest_y[11] ), .A2(new_n122), .B(\dest_y[12] ), .Y(new_n123));
  OR2x2_ASAP7_75t_R         g33(.A(\dest_y[13] ), .B(new_n123), .Y(new_n124));
  AOI31xp33_ASAP7_75t_R     g34(.A1(\dest_y[14] ), .A2(\dest_y[15] ), .A3(new_n124), .B(\dest_y[16] ), .Y(new_n125));
  INVx1_ASAP7_75t_R         g35(.A(\dest_y[13] ), .Y(new_n126));
  NAND2xp33_ASAP7_75t_R     g36(.A(\dest_y[19] ), .B(\dest_y[20] ), .Y(new_n127));
  INVx1_ASAP7_75t_R         g37(.A(\dest_y[26] ), .Y(new_n128));
  NAND4xp25_ASAP7_75t_R     g38(.A(\dest_y[23] ), .B(\dest_y[24] ), .C(\dest_y[25] ), .D(new_n128), .Y(new_n129));
  INVx1_ASAP7_75t_R         g39(.A(\dest_y[11] ), .Y(new_n130));
  OAI22xp33_ASAP7_75t_R     g40(.A1(new_n130), .A2(new_n121), .B1(\dest_y[11] ), .B2(new_n122), .Y(new_n131));
  NOR5xp2_ASAP7_75t_R       g41(.A(\dest_y[21] ), .B(\dest_y[22] ), .C(new_n127), .D(new_n129), .E(new_n131), .Y(new_n132));
  INVx1_ASAP7_75t_R         g42(.A(\dest_y[28] ), .Y(new_n133));
  INVx1_ASAP7_75t_R         g43(.A(\dest_y[29] ), .Y(new_n134));
  INVx1_ASAP7_75t_R         g44(.A(\dest_y[27] ), .Y(new_n135));
  OAI22xp33_ASAP7_75t_R     g45(.A1(new_n128), .A2(new_n135), .B1(\dest_y[26] ), .B2(\dest_y[27] ), .Y(new_n136));
  INVx1_ASAP7_75t_R         g46(.A(\dest_y[0] ), .Y(new_n137));
  NOR3xp33_ASAP7_75t_R      g47(.A(\dest_x[0] ), .B(new_n137), .C(new_n122), .Y(new_n138));
  NAND5xp2_ASAP7_75t_R      g48(.A(\dest_y[1] ), .B(\dest_y[2] ), .C(\dest_y[3] ), .D(\dest_y[4] ), .E(new_n138), .Y(new_n139));
  INVx1_ASAP7_75t_R         g49(.A(\dest_y[15] ), .Y(new_n140));
  NOR4xp25_ASAP7_75t_R      g50(.A(\dest_y[12] ), .B(new_n140), .C(new_n120), .D(\dest_y[18] ), .Y(new_n141));
  NAND5xp2_ASAP7_75t_R      g51(.A(\dest_y[5] ), .B(\dest_y[6] ), .C(\dest_y[7] ), .D(\dest_y[8] ), .E(new_n141), .Y(new_n142));
  NOR5xp2_ASAP7_75t_R       g52(.A(new_n133), .B(new_n134), .C(new_n136), .D(new_n139), .E(new_n142), .Y(new_n143));
  OAI211xp5_ASAP7_75t_R     g53(.A1(\dest_y[14] ), .A2(new_n123), .B(new_n132), .C(new_n143), .Y(new_n144));
  O2A1O1Ixp33_ASAP7_75t_R   g54(.A1(new_n126), .A2(\dest_y[14] ), .B(new_n124), .C(new_n144), .Y(new_n145));
  NOR4xp25_ASAP7_75t_R      g55(.A(new_n133), .B(new_n134), .C(new_n128), .D(new_n135), .Y(new_n146));
  O2A1O1Ixp33_ASAP7_75t_R   g56(.A1(new_n120), .A2(new_n125), .B(new_n145), .C(new_n146), .Y(new_n147));
  INVx1_ASAP7_75t_R         g57(.A(new_n118), .Y(new_n148));
  AOI31xp33_ASAP7_75t_R     g58(.A1(new_n99), .A2(new_n147), .A3(new_n148), .B(new_n100), .Y(new_n149));
  INVx1_ASAP7_75t_R         g59(.A(new_n149), .Y(\outport[1] ));
  NOR5xp2_ASAP7_75t_R       g60(.A(new_n133), .B(new_n134), .C(new_n128), .D(new_n135), .E(\outport[0] ), .Y(\outport[2] ));
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


