// Benchmark "cavlc" written by ABC on Thu Apr  2 14:52:08 2026

module cavlc ( 
    \totalcoeffs[0] , \totalcoeffs[1] , \totalcoeffs[2] , \totalcoeffs[3] ,
    \totalcoeffs[4] , \ctable[0] , \ctable[1] , \ctable[2] ,
    \trailingones[0] , \trailingones[1] ,
    \coeff_token[0] , \coeff_token[1] , \coeff_token[2] , \coeff_token[3] ,
    \coeff_token[4] , \coeff_token[5] , \ctoken_len[0] , \ctoken_len[1] ,
    \ctoken_len[2] , \ctoken_len[3] , \ctoken_len[4]   );
  input  \totalcoeffs[0] , \totalcoeffs[1] , \totalcoeffs[2] ,
    \totalcoeffs[3] , \totalcoeffs[4] , \ctable[0] , \ctable[1] ,
    \ctable[2] , \trailingones[0] , \trailingones[1] ;
  output \coeff_token[0] , \coeff_token[1] , \coeff_token[2] ,
    \coeff_token[3] , \coeff_token[4] , \coeff_token[5] , \ctoken_len[0] ,
    \ctoken_len[1] , \ctoken_len[2] , \ctoken_len[3] , \ctoken_len[4] ;
  wire new_n22, new_n23, new_n24, new_n25, new_n26, new_n27, new_n28,
    new_n29, new_n30, new_n31, new_n32, new_n33, new_n34, new_n35, new_n36,
    new_n37, new_n38, new_n39, new_n40, new_n41, new_n42, new_n43, new_n44,
    new_n45, new_n46, new_n47, new_n48, new_n49, new_n50, new_n51, new_n52,
    new_n53, new_n54, new_n55, new_n56, new_n57, new_n58, new_n59, new_n60,
    new_n61, new_n62, new_n63, new_n64, new_n65, new_n66, new_n67, new_n68,
    new_n69, new_n70, new_n71, new_n72, new_n73, new_n74, new_n75, new_n76,
    new_n77, new_n78, new_n79, new_n81, new_n82, new_n83, new_n84, new_n85,
    new_n86, new_n87, new_n88, new_n89, new_n90, new_n91, new_n92, new_n93,
    new_n94, new_n95, new_n96, new_n97, new_n98, new_n99, new_n100,
    new_n101, new_n102, new_n103, new_n104, new_n105, new_n106, new_n107,
    new_n108, new_n109, new_n110, new_n111, new_n112, new_n113, new_n114,
    new_n115, new_n116, new_n117, new_n118, new_n119, new_n120, new_n121,
    new_n122, new_n123, new_n124, new_n125, new_n126, new_n127, new_n128,
    new_n129, new_n130, new_n131, new_n132, new_n133, new_n134, new_n135,
    new_n136, new_n137, new_n138, new_n139, new_n141, new_n142, new_n143,
    new_n144, new_n145, new_n146, new_n147, new_n148, new_n149, new_n150,
    new_n151, new_n152, new_n153, new_n154, new_n155, new_n156, new_n157,
    new_n158, new_n159, new_n160, new_n161, new_n162, new_n163, new_n164,
    new_n165, new_n166, new_n167, new_n168, new_n169, new_n170, new_n171,
    new_n172, new_n173, new_n174, new_n175, new_n176, new_n177, new_n178,
    new_n180, new_n181, new_n182, new_n183, new_n184, new_n185, new_n186,
    new_n187, new_n188, new_n189, new_n190, new_n191, new_n192, new_n193,
    new_n194, new_n195, new_n196, new_n197, new_n198, new_n199, new_n200,
    new_n201, new_n202, new_n203, new_n204, new_n205, new_n206, new_n207,
    new_n208, new_n209, new_n210, new_n211, new_n213, new_n214, new_n215,
    new_n216, new_n217, new_n219, new_n220, new_n222, new_n223, new_n224,
    new_n225, new_n226, new_n227, new_n228, new_n229, new_n230, new_n231,
    new_n232, new_n233, new_n234, new_n235, new_n236, new_n237, new_n238,
    new_n239, new_n240, new_n241, new_n242, new_n243, new_n244, new_n245,
    new_n246, new_n247, new_n248, new_n249, new_n250, new_n251, new_n252,
    new_n254, new_n255, new_n256, new_n257, new_n258, new_n259, new_n260,
    new_n261, new_n262, new_n263, new_n264, new_n265, new_n266, new_n267,
    new_n268, new_n269, new_n270, new_n271, new_n272, new_n273, new_n274,
    new_n275, new_n276, new_n277, new_n278, new_n279, new_n280, new_n281,
    new_n282, new_n283, new_n284, new_n285, new_n286, new_n287, new_n289,
    new_n290, new_n291, new_n292, new_n293, new_n294, new_n295, new_n296,
    new_n297, new_n298, new_n299, new_n300, new_n301, new_n302, new_n303,
    new_n304, new_n305, new_n306, new_n307, new_n308, new_n309, new_n310,
    new_n311, new_n313, new_n314, new_n315, new_n316, new_n317, new_n318,
    new_n319, new_n320, new_n321, new_n322, new_n323, new_n324, new_n326,
    new_n327;
  INVx1_ASAP7_75t_R         g000(.A(\ctable[1] ), .Y(new_n22));
  INVx1_ASAP7_75t_R         g001(.A(\totalcoeffs[0] ), .Y(new_n23));
  INVx1_ASAP7_75t_R         g002(.A(\totalcoeffs[2] ), .Y(new_n24));
  INVx1_ASAP7_75t_R         g003(.A(\totalcoeffs[3] ), .Y(new_n25));
  NAND2xp33_ASAP7_75t_R     g004(.A(new_n24), .B(new_n25), .Y(new_n26));
  INVx1_ASAP7_75t_R         g005(.A(\totalcoeffs[1] ), .Y(new_n27));
  NOR2xp33_ASAP7_75t_R      g006(.A(\ctable[0] ), .B(\trailingones[1] ), .Y(new_n28));
  INVx1_ASAP7_75t_R         g007(.A(\ctable[2] ), .Y(new_n29));
  INVx1_ASAP7_75t_R         g008(.A(\trailingones[1] ), .Y(new_n30));
  INVx1_ASAP7_75t_R         g009(.A(\trailingones[0] ), .Y(new_n31));
  NOR2xp33_ASAP7_75t_R      g010(.A(\ctable[0] ), .B(new_n31), .Y(new_n32));
  INVx1_ASAP7_75t_R         g011(.A(new_n32), .Y(new_n33));
  AOI221xp5_ASAP7_75t_R     g012(.A1(new_n29), .A2(new_n30), .B1(\ctable[2] ), .B2(new_n33), .C(new_n27), .Y(new_n34));
  O2A1O1Ixp33_ASAP7_75t_R   g013(.A1(new_n27), .A2(\ctable[2] ), .B(new_n28), .C(new_n34), .Y(new_n35));
  NOR2xp33_ASAP7_75t_R      g014(.A(\totalcoeffs[0] ), .B(new_n24), .Y(new_n36));
  NAND2xp33_ASAP7_75t_R     g015(.A(new_n27), .B(new_n25), .Y(new_n37));
  NOR2xp33_ASAP7_75t_R      g016(.A(new_n24), .B(new_n30), .Y(new_n38));
  INVx1_ASAP7_75t_R         g017(.A(new_n38), .Y(new_n39));
  OAI21xp33_ASAP7_75t_R     g018(.A1(\totalcoeffs[2] ), .A2(\trailingones[1] ), .B(new_n39), .Y(new_n40));
  NOR3xp33_ASAP7_75t_R      g019(.A(new_n36), .B(new_n37), .C(new_n40), .Y(new_n41));
  NOR2xp33_ASAP7_75t_R      g020(.A(new_n24), .B(\trailingones[1] ), .Y(new_n42));
  NOR2xp33_ASAP7_75t_R      g021(.A(new_n23), .B(\trailingones[1] ), .Y(new_n43));
  NOR2xp33_ASAP7_75t_R      g022(.A(\totalcoeffs[2] ), .B(new_n43), .Y(new_n44));
  NOR4xp25_ASAP7_75t_R      g023(.A(new_n27), .B(new_n25), .C(new_n42), .D(new_n44), .Y(new_n45));
  OR3x1_ASAP7_75t_R         g024(.A(\ctable[2] ), .B(\trailingones[0] ), .C(new_n45), .Y(new_n46));
  OAI32xp33_ASAP7_75t_R     g025(.A1(new_n23), .A2(new_n26), .A3(new_n35), .B1(new_n41), .B2(new_n46), .Y(new_n47));
  NOR2xp33_ASAP7_75t_R      g026(.A(\totalcoeffs[0] ), .B(new_n30), .Y(new_n48));
  INVx1_ASAP7_75t_R         g027(.A(new_n43), .Y(new_n49));
  NOR3xp33_ASAP7_75t_R      g028(.A(\totalcoeffs[1] ), .B(new_n24), .C(new_n49), .Y(new_n50));
  O2A1O1Ixp33_ASAP7_75t_R   g029(.A1(\totalcoeffs[1] ), .A2(new_n48), .B(new_n31), .C(new_n50), .Y(new_n51));
  INVx1_ASAP7_75t_R         g030(.A(\ctable[0] ), .Y(new_n52));
  NOR2xp33_ASAP7_75t_R      g031(.A(new_n52), .B(new_n22), .Y(new_n53));
  INVx1_ASAP7_75t_R         g032(.A(new_n53), .Y(new_n54));
  OAI22xp33_ASAP7_75t_R     g033(.A1(\ctable[0] ), .A2(new_n51), .B1(new_n31), .B2(new_n54), .Y(new_n55));
  INVx1_ASAP7_75t_R         g034(.A(new_n55), .Y(new_n56));
  NOR2xp33_ASAP7_75t_R      g035(.A(\totalcoeffs[1] ), .B(\trailingones[1] ), .Y(new_n57));
  NOR2xp33_ASAP7_75t_R      g036(.A(\totalcoeffs[0] ), .B(new_n26), .Y(new_n58));
  NOR2xp33_ASAP7_75t_R      g037(.A(new_n24), .B(\totalcoeffs[3] ), .Y(new_n59));
  INVx1_ASAP7_75t_R         g038(.A(new_n59), .Y(new_n60));
  OAI32xp33_ASAP7_75t_R     g039(.A1(\totalcoeffs[1] ), .A2(new_n30), .A3(new_n60), .B1(new_n27), .B2(\trailingones[0] ), .Y(new_n61));
  NAND2xp33_ASAP7_75t_R     g040(.A(\totalcoeffs[1] ), .B(new_n25), .Y(new_n62));
  INVx1_ASAP7_75t_R         g041(.A(new_n48), .Y(new_n63));
  NAND2xp33_ASAP7_75t_R     g042(.A(new_n49), .B(new_n63), .Y(new_n64));
  OAI32xp33_ASAP7_75t_R     g043(.A1(\totalcoeffs[2] ), .A2(new_n62), .A3(new_n64), .B1(new_n22), .B2(new_n44), .Y(new_n65));
  NOR2xp33_ASAP7_75t_R      g044(.A(new_n52), .B(new_n31), .Y(new_n66));
  AOI332xp33_ASAP7_75t_R    g045(.A1(new_n31), .A2(new_n57), .A3(new_n58), .B1(new_n23), .B2(new_n52), .B3(new_n61), .C1(new_n65), .C2(new_n66), .Y(new_n67));
  OAI21xp33_ASAP7_75t_R     g046(.A1(new_n25), .A2(new_n56), .B(new_n67), .Y(new_n68));
  AOI22xp33_ASAP7_75t_R     g047(.A1(new_n22), .A2(new_n47), .B1(new_n29), .B2(new_n68), .Y(new_n69));
  INVx1_ASAP7_75t_R         g048(.A(\totalcoeffs[4] ), .Y(new_n70));
  NOR2xp33_ASAP7_75t_R      g049(.A(new_n70), .B(new_n30), .Y(new_n71));
  NOR2xp33_ASAP7_75t_R      g050(.A(\ctable[1] ), .B(\trailingones[1] ), .Y(new_n72));
  INVx1_ASAP7_75t_R         g051(.A(new_n72), .Y(new_n73));
  NAND2xp33_ASAP7_75t_R     g052(.A(new_n22), .B(\trailingones[0] ), .Y(new_n74));
  NAND2xp33_ASAP7_75t_R     g053(.A(\ctable[0] ), .B(new_n74), .Y(new_n75));
  O2A1O1Ixp33_ASAP7_75t_R   g054(.A1(new_n70), .A2(new_n31), .B(new_n73), .C(new_n75), .Y(new_n76));
  AOI31xp33_ASAP7_75t_R     g055(.A1(new_n31), .A2(new_n54), .A3(new_n71), .B(new_n76), .Y(new_n77));
  NOR2xp33_ASAP7_75t_R      g056(.A(\totalcoeffs[0] ), .B(\totalcoeffs[2] ), .Y(new_n78));
  NAND4xp25_ASAP7_75t_R     g057(.A(new_n27), .B(new_n25), .C(new_n29), .D(new_n78), .Y(new_n79));
  OAI22xp33_ASAP7_75t_R     g058(.A1(\totalcoeffs[4] ), .A2(new_n69), .B1(new_n77), .B2(new_n79), .Y(\coeff_token[0] ));
  NAND2xp33_ASAP7_75t_R     g059(.A(new_n31), .B(\trailingones[1] ), .Y(new_n81));
  NOR2xp33_ASAP7_75t_R      g060(.A(new_n22), .B(new_n31), .Y(new_n82));
  AOI21xp33_ASAP7_75t_R     g061(.A1(\ctable[1] ), .A2(\trailingones[1] ), .B(new_n32), .Y(new_n83));
  OA33x2_ASAP7_75t_R        g062(.A1(\totalcoeffs[0] ), .A2(new_n52), .A3(new_n81), .B1(new_n23), .B2(new_n82), .B3(new_n83), .Y(new_n84));
  NOR2xp33_ASAP7_75t_R      g063(.A(\ctable[1] ), .B(\trailingones[0] ), .Y(new_n85));
  O2A1O1Ixp33_ASAP7_75t_R   g064(.A1(\totalcoeffs[0] ), .A2(new_n22), .B(new_n24), .C(new_n33), .Y(new_n86));
  OAI21xp33_ASAP7_75t_R     g065(.A1(new_n85), .A2(new_n86), .B(new_n30), .Y(new_n87));
  OAI21xp33_ASAP7_75t_R     g066(.A1(\totalcoeffs[2] ), .A2(new_n84), .B(new_n87), .Y(new_n88));
  AO32x1_ASAP7_75t_R        g067(.A1(new_n22), .A2(\trailingones[0] ), .A3(new_n42), .B1(\totalcoeffs[1] ), .B2(new_n88), .Y(new_n89));
  NOR2xp33_ASAP7_75t_R      g068(.A(new_n27), .B(new_n24), .Y(new_n90));
  INVx1_ASAP7_75t_R         g069(.A(new_n90), .Y(new_n91));
  NAND2xp33_ASAP7_75t_R     g070(.A(\ctable[0] ), .B(new_n31), .Y(new_n92));
  NOR2xp33_ASAP7_75t_R      g071(.A(\totalcoeffs[2] ), .B(\trailingones[0] ), .Y(new_n93));
  INVx1_ASAP7_75t_R         g072(.A(new_n93), .Y(new_n94));
  NAND2xp33_ASAP7_75t_R     g073(.A(new_n27), .B(\totalcoeffs[3] ), .Y(new_n95));
  NAND2xp33_ASAP7_75t_R     g074(.A(\totalcoeffs[1] ), .B(\ctable[0] ), .Y(new_n96));
  O2A1O1Ixp33_ASAP7_75t_R   g075(.A1(new_n94), .A2(new_n95), .B(new_n96), .C(new_n23), .Y(new_n97));
  A2O1A1O1Ixp25_ASAP7_75t_R g076(.A1(\totalcoeffs[3] ), .A2(new_n90), .B(\ctable[0] ), .C(new_n26), .D(new_n97), .Y(new_n98));
  OAI32xp33_ASAP7_75t_R     g077(.A1(new_n25), .A2(new_n91), .A3(new_n92), .B1(new_n22), .B2(new_n98), .Y(new_n99));
  AOI21xp33_ASAP7_75t_R     g078(.A1(\totalcoeffs[3] ), .A2(\trailingones[0] ), .B(\totalcoeffs[2] ), .Y(new_n100));
  NOR2xp33_ASAP7_75t_R      g079(.A(\totalcoeffs[3] ), .B(\trailingones[0] ), .Y(new_n101));
  NOR2xp33_ASAP7_75t_R      g080(.A(new_n32), .B(new_n101), .Y(new_n102));
  AOI22xp33_ASAP7_75t_R     g081(.A1(\ctable[0] ), .A2(new_n93), .B1(\totalcoeffs[0] ), .B2(new_n102), .Y(new_n103));
  NOR2xp33_ASAP7_75t_R      g082(.A(\totalcoeffs[0] ), .B(new_n52), .Y(new_n104));
  NOR2xp33_ASAP7_75t_R      g083(.A(new_n23), .B(\ctable[0] ), .Y(new_n105));
  NOR2xp33_ASAP7_75t_R      g084(.A(\totalcoeffs[2] ), .B(new_n31), .Y(new_n106));
  INVx1_ASAP7_75t_R         g085(.A(new_n106), .Y(new_n107));
  NOR2xp33_ASAP7_75t_R      g086(.A(new_n23), .B(new_n107), .Y(new_n108));
  OAI31xp33_ASAP7_75t_R     g087(.A1(new_n104), .A2(new_n105), .A3(new_n108), .B(\totalcoeffs[1] ), .Y(new_n109));
  OAI221xp5_ASAP7_75t_R     g088(.A1(\totalcoeffs[0] ), .A2(new_n100), .B1(\totalcoeffs[1] ), .B2(new_n103), .C(new_n109), .Y(new_n110));
  NOR2xp33_ASAP7_75t_R      g089(.A(new_n23), .B(new_n27), .Y(new_n111));
  NOR2xp33_ASAP7_75t_R      g090(.A(\totalcoeffs[0] ), .B(new_n31), .Y(new_n112));
  NAND2xp33_ASAP7_75t_R     g091(.A(new_n27), .B(new_n22), .Y(new_n113));
  NAND2xp33_ASAP7_75t_R     g092(.A(new_n24), .B(new_n113), .Y(new_n114));
  NOR2xp33_ASAP7_75t_R      g093(.A(\totalcoeffs[1] ), .B(\trailingones[0] ), .Y(new_n115));
  AOI21xp33_ASAP7_75t_R     g094(.A1(new_n27), .A2(\ctable[1] ), .B(new_n24), .Y(new_n116));
  OAI33xp33_ASAP7_75t_R     g095(.A1(new_n111), .A2(new_n112), .A3(new_n114), .B1(new_n25), .B2(new_n115), .B3(new_n116), .Y(new_n117));
  AOI22xp33_ASAP7_75t_R     g096(.A1(new_n22), .A2(new_n110), .B1(new_n52), .B2(new_n117), .Y(new_n118));
  NOR2xp33_ASAP7_75t_R      g097(.A(\trailingones[1] ), .B(new_n118), .Y(new_n119));
  AOI221xp5_ASAP7_75t_R     g098(.A1(new_n25), .A2(new_n89), .B1(\trailingones[1] ), .B2(new_n99), .C(new_n119), .Y(new_n120));
  OAI332xp33_ASAP7_75t_R    g099(.A1(\ctable[0] ), .A2(\trailingones[1] ), .A3(\trailingones[0] ), .B1(new_n22), .B2(new_n31), .B3(new_n30), .C1(\totalcoeffs[0] ), .C2(new_n83), .Y(new_n121));
  NOR2xp33_ASAP7_75t_R      g100(.A(\totalcoeffs[2] ), .B(\trailingones[1] ), .Y(new_n122));
  NOR2xp33_ASAP7_75t_R      g101(.A(new_n23), .B(new_n31), .Y(new_n123));
  NOR2xp33_ASAP7_75t_R      g102(.A(new_n122), .B(new_n123), .Y(new_n124));
  NOR4xp25_ASAP7_75t_R      g103(.A(new_n52), .B(new_n43), .C(new_n106), .D(new_n124), .Y(new_n125));
  O2A1O1Ixp33_ASAP7_75t_R   g104(.A1(new_n70), .A2(new_n31), .B(new_n92), .C(\ctable[1] ), .Y(new_n126));
  OAI221xp5_ASAP7_75t_R     g105(.A1(\ctable[1] ), .A2(new_n30), .B1(new_n71), .B2(new_n126), .C(new_n78), .Y(new_n127));
  INVx1_ASAP7_75t_R         g106(.A(new_n127), .Y(new_n128));
  A2O1A1O1Ixp25_ASAP7_75t_R g107(.A1(\totalcoeffs[2] ), .A2(new_n121), .B(new_n125), .C(new_n70), .D(new_n128), .Y(new_n129));
  OAI22xp33_ASAP7_75t_R     g108(.A1(\totalcoeffs[4] ), .A2(new_n120), .B1(new_n37), .B2(new_n129), .Y(new_n130));
  INVx1_ASAP7_75t_R         g109(.A(new_n130), .Y(new_n131));
  NAND3xp33_ASAP7_75t_R     g110(.A(new_n25), .B(new_n52), .C(new_n22), .Y(new_n132));
  NAND2xp33_ASAP7_75t_R     g111(.A(new_n23), .B(new_n27), .Y(new_n133));
  NAND2xp33_ASAP7_75t_R     g112(.A(\totalcoeffs[2] ), .B(new_n133), .Y(new_n134));
  INVx1_ASAP7_75t_R         g113(.A(new_n111), .Y(new_n135));
  NAND2xp33_ASAP7_75t_R     g114(.A(new_n44), .B(new_n135), .Y(new_n136));
  NOR2xp33_ASAP7_75t_R      g115(.A(new_n27), .B(\trailingones[1] ), .Y(new_n137));
  AOI32xp33_ASAP7_75t_R     g116(.A1(new_n31), .A2(new_n134), .A3(new_n136), .B1(new_n106), .B2(new_n137), .Y(new_n138));
  OR4x1_ASAP7_75t_R         g117(.A(\totalcoeffs[4] ), .B(new_n29), .C(new_n132), .D(new_n138), .Y(new_n139));
  OAI21xp33_ASAP7_75t_R     g118(.A1(\ctable[2] ), .A2(new_n131), .B(new_n139), .Y(\coeff_token[1] ));
  NOR2xp33_ASAP7_75t_R      g119(.A(new_n31), .B(\trailingones[1] ), .Y(new_n141));
  NOR2xp33_ASAP7_75t_R      g120(.A(\totalcoeffs[2] ), .B(new_n37), .Y(new_n142));
  OAI311xp33_ASAP7_75t_R    g121(.A1(\ctable[0] ), .A2(new_n85), .A3(new_n141), .B1(new_n23), .C1(new_n142), .Y(new_n143));
  INVx1_ASAP7_75t_R         g122(.A(new_n112), .Y(new_n144));
  INVx1_ASAP7_75t_R         g123(.A(new_n141), .Y(new_n145));
  NAND2xp33_ASAP7_75t_R     g124(.A(new_n81), .B(new_n145), .Y(new_n146));
  INVx1_ASAP7_75t_R         g125(.A(new_n146), .Y(new_n147));
  AOI32xp33_ASAP7_75t_R     g126(.A1(\ctable[0] ), .A2(new_n144), .A3(new_n147), .B1(new_n52), .B2(new_n38), .Y(new_n148));
  OAI21xp33_ASAP7_75t_R     g127(.A1(new_n57), .A2(new_n146), .B(\totalcoeffs[2] ), .Y(new_n149));
  O2A1O1Ixp33_ASAP7_75t_R   g128(.A1(new_n27), .A2(new_n148), .B(new_n149), .C(\totalcoeffs[3] ), .Y(new_n150));
  AOI211xp5_ASAP7_75t_R     g129(.A1(\ctable[0] ), .A2(new_n95), .B(new_n49), .C(new_n94), .Y(new_n151));
  INVx1_ASAP7_75t_R         g130(.A(new_n105), .Y(new_n152));
  NOR2xp33_ASAP7_75t_R      g131(.A(new_n27), .B(new_n30), .Y(new_n153));
  INVx1_ASAP7_75t_R         g132(.A(new_n153), .Y(new_n154));
  OAI21xp33_ASAP7_75t_R     g133(.A1(\totalcoeffs[1] ), .A2(\trailingones[1] ), .B(new_n154), .Y(new_n155));
  AOI22xp33_ASAP7_75t_R     g134(.A1(new_n101), .A2(new_n155), .B1(\totalcoeffs[1] ), .B2(new_n141), .Y(new_n156));
  NOR2xp33_ASAP7_75t_R      g135(.A(new_n22), .B(new_n115), .Y(new_n157));
  NOR2xp33_ASAP7_75t_R      g136(.A(\trailingones[0] ), .B(\trailingones[1] ), .Y(new_n158));
  NAND2xp33_ASAP7_75t_R     g137(.A(\totalcoeffs[2] ), .B(new_n158), .Y(new_n159));
  OAI21xp33_ASAP7_75t_R     g138(.A1(new_n31), .A2(new_n42), .B(new_n159), .Y(new_n160));
  O2A1O1Ixp33_ASAP7_75t_R   g139(.A1(new_n27), .A2(new_n38), .B(new_n157), .C(new_n160), .Y(new_n161));
  OAI22xp33_ASAP7_75t_R     g140(.A1(\totalcoeffs[2] ), .A2(new_n156), .B1(new_n25), .B2(new_n161), .Y(new_n162));
  INVx1_ASAP7_75t_R         g141(.A(new_n162), .Y(new_n163));
  NAND3xp33_ASAP7_75t_R     g142(.A(new_n52), .B(\ctable[1] ), .C(new_n25), .Y(new_n164));
  O2A1O1Ixp33_ASAP7_75t_R   g143(.A1(new_n24), .A2(\ctable[0] ), .B(\totalcoeffs[3] ), .C(new_n31), .Y(new_n165));
  O2A1O1Ixp33_ASAP7_75t_R   g144(.A1(new_n25), .A2(new_n106), .B(\ctable[0] ), .C(new_n165), .Y(new_n166));
  OAI32xp33_ASAP7_75t_R     g145(.A1(\totalcoeffs[1] ), .A2(\trailingones[0] ), .A3(new_n164), .B1(new_n27), .B2(new_n166), .Y(new_n167));
  AOI21xp33_ASAP7_75t_R     g146(.A1(\totalcoeffs[2] ), .A2(new_n74), .B(new_n27), .Y(new_n168));
  AOI211xp5_ASAP7_75t_R     g147(.A1(\totalcoeffs[1] ), .A2(\trailingones[0] ), .B(\totalcoeffs[2] ), .C(new_n22), .Y(new_n169));
  O2A1O1Ixp33_ASAP7_75t_R   g148(.A1(new_n24), .A2(new_n115), .B(\trailingones[1] ), .C(new_n169), .Y(new_n170));
  NOR2xp33_ASAP7_75t_R      g149(.A(\totalcoeffs[3] ), .B(new_n30), .Y(new_n171));
  NOR2xp33_ASAP7_75t_R      g150(.A(new_n22), .B(\trailingones[0] ), .Y(new_n172));
  AOI211xp5_ASAP7_75t_R     g151(.A1(new_n62), .A2(new_n81), .B(\ctable[1] ), .C(new_n171), .Y(new_n173));
  AOI31xp33_ASAP7_75t_R     g152(.A1(\totalcoeffs[1] ), .A2(new_n171), .A3(new_n172), .B(new_n173), .Y(new_n174));
  OAI321xp33_ASAP7_75t_R    g153(.A1(new_n52), .A2(new_n100), .A3(new_n168), .B1(new_n25), .B2(new_n170), .C(new_n174), .Y(new_n175));
  AOI21xp33_ASAP7_75t_R     g154(.A1(new_n30), .A2(new_n167), .B(new_n175), .Y(new_n176));
  OAI221xp5_ASAP7_75t_R     g155(.A1(new_n152), .A2(new_n163), .B1(\totalcoeffs[0] ), .B2(new_n176), .C(new_n70), .Y(new_n177));
  O2A1O1Ixp33_ASAP7_75t_R   g156(.A1(new_n150), .A2(new_n151), .B(new_n22), .C(new_n177), .Y(new_n178));
  AOI211xp5_ASAP7_75t_R     g157(.A1(\totalcoeffs[4] ), .A2(new_n143), .B(\ctable[2] ), .C(new_n178), .Y(\coeff_token[2] ));
  NAND2xp33_ASAP7_75t_R     g158(.A(\ctable[0] ), .B(new_n93), .Y(new_n180));
  O2A1O1Ixp33_ASAP7_75t_R   g159(.A1(new_n180), .A2(new_n113), .B(new_n164), .C(\trailingones[1] ), .Y(new_n181));
  OAI21xp33_ASAP7_75t_R     g160(.A1(new_n52), .A2(new_n81), .B(new_n25), .Y(new_n182));
  NAND2xp33_ASAP7_75t_R     g161(.A(new_n24), .B(\ctable[0] ), .Y(new_n183));
  OAI22xp33_ASAP7_75t_R     g162(.A1(new_n24), .A2(new_n102), .B1(new_n31), .B2(new_n183), .Y(new_n184));
  NOR2xp33_ASAP7_75t_R      g163(.A(new_n25), .B(\ctable[0] ), .Y(new_n185));
  O2A1O1Ixp33_ASAP7_75t_R   g164(.A1(new_n25), .A2(\trailingones[0] ), .B(new_n22), .C(new_n185), .Y(new_n186));
  AOI221xp5_ASAP7_75t_R     g165(.A1(new_n24), .A2(new_n182), .B1(new_n30), .B2(new_n184), .C(new_n186), .Y(new_n187));
  NOR2xp33_ASAP7_75t_R      g166(.A(new_n27), .B(new_n187), .Y(new_n188));
  NOR3xp33_ASAP7_75t_R      g167(.A(\ctable[0] ), .B(new_n22), .C(\trailingones[1] ), .Y(new_n189));
  INVx1_ASAP7_75t_R         g168(.A(new_n189), .Y(new_n190));
  AOI21xp33_ASAP7_75t_R     g169(.A1(\ctable[0] ), .A2(\trailingones[1] ), .B(\ctable[1] ), .Y(new_n191));
  OA21x2_ASAP7_75t_R        g170(.A1(new_n66), .A2(new_n153), .B(new_n191), .Y(new_n192));
  O2A1O1Ixp33_ASAP7_75t_R   g171(.A1(new_n28), .A2(new_n85), .B(new_n24), .C(new_n192), .Y(new_n193));
  NOR2xp33_ASAP7_75t_R      g172(.A(new_n25), .B(\ctable[1] ), .Y(new_n194));
  INVx1_ASAP7_75t_R         g173(.A(new_n194), .Y(new_n195));
  NOR2xp33_ASAP7_75t_R      g174(.A(new_n24), .B(new_n195), .Y(new_n196));
  INVx1_ASAP7_75t_R         g175(.A(new_n196), .Y(new_n197));
  NOR2xp33_ASAP7_75t_R      g176(.A(new_n30), .B(new_n197), .Y(new_n198));
  O2A1O1Ixp33_ASAP7_75t_R   g177(.A1(new_n189), .A2(new_n194), .B(new_n31), .C(new_n198), .Y(new_n199));
  OAI322xp33_ASAP7_75t_R    g178(.A1(new_n27), .A2(new_n31), .A3(new_n190), .B1(new_n25), .B2(new_n193), .C1(\totalcoeffs[1] ), .C2(new_n199), .Y(new_n200));
  O2A1O1Ixp33_ASAP7_75t_R   g179(.A1(new_n181), .A2(new_n188), .B(\totalcoeffs[0] ), .C(new_n200), .Y(new_n201));
  AOI22xp33_ASAP7_75t_R     g180(.A1(\totalcoeffs[3] ), .A2(new_n30), .B1(\ctable[1] ), .B2(new_n26), .Y(new_n202));
  INVx1_ASAP7_75t_R         g181(.A(new_n158), .Y(new_n203));
  O2A1O1Ixp33_ASAP7_75t_R   g182(.A1(new_n52), .A2(new_n30), .B(new_n203), .C(new_n24), .Y(new_n204));
  OAI311xp33_ASAP7_75t_R    g183(.A1(\totalcoeffs[3] ), .A2(new_n172), .A3(new_n204), .B1(\totalcoeffs[1] ), .C1(new_n75), .Y(new_n205));
  OAI321xp33_ASAP7_75t_R    g184(.A1(\ctable[0] ), .A2(new_n22), .A3(new_n24), .B1(\totalcoeffs[1] ), .B2(new_n202), .C(new_n205), .Y(new_n206));
  NOR2xp33_ASAP7_75t_R      g185(.A(\ctable[0] ), .B(\ctable[1] ), .Y(new_n207));
  NOR4xp25_ASAP7_75t_R      g186(.A(\totalcoeffs[3] ), .B(new_n70), .C(\totalcoeffs[1] ), .D(\totalcoeffs[2] ), .Y(new_n208));
  OAI21xp33_ASAP7_75t_R     g187(.A1(new_n31), .A2(new_n30), .B(new_n52), .Y(new_n209));
  OA211x2_ASAP7_75t_R       g188(.A1(new_n53), .A2(new_n207), .B(new_n208), .C(new_n209), .Y(new_n210));
  A2O1A1Ixp33_ASAP7_75t_R   g189(.A1(new_n70), .A2(new_n206), .B(new_n210), .C(new_n23), .Y(new_n211));
  O2A1O1Ixp33_ASAP7_75t_R   g190(.A1(\totalcoeffs[4] ), .A2(new_n201), .B(new_n211), .C(\ctable[2] ), .Y(\coeff_token[3] ));
  NOR2xp33_ASAP7_75t_R      g191(.A(\totalcoeffs[3] ), .B(new_n70), .Y(new_n213));
  NOR2xp33_ASAP7_75t_R      g192(.A(new_n25), .B(\totalcoeffs[4] ), .Y(new_n214));
  NOR2xp33_ASAP7_75t_R      g193(.A(\totalcoeffs[2] ), .B(new_n133), .Y(new_n215));
  OAI21xp33_ASAP7_75t_R     g194(.A1(new_n213), .A2(new_n214), .B(new_n215), .Y(new_n216));
  NAND2xp33_ASAP7_75t_R     g195(.A(new_n29), .B(new_n53), .Y(new_n217));
  O2A1O1Ixp33_ASAP7_75t_R   g196(.A1(\totalcoeffs[4] ), .A2(new_n134), .B(new_n216), .C(new_n217), .Y(\coeff_token[4] ));
  INVx1_ASAP7_75t_R         g197(.A(new_n214), .Y(new_n219));
  NAND2xp33_ASAP7_75t_R     g198(.A(new_n213), .B(new_n215), .Y(new_n220));
  O2A1O1Ixp33_ASAP7_75t_R   g199(.A1(new_n219), .A2(new_n215), .B(new_n220), .C(new_n217), .Y(\coeff_token[5] ));
  OAI31xp33_ASAP7_75t_R     g200(.A1(new_n70), .A2(new_n26), .A3(new_n133), .B(new_n54), .Y(new_n222));
  INVx1_ASAP7_75t_R         g201(.A(new_n36), .Y(new_n223));
  A2O1A1Ixp33_ASAP7_75t_R   g202(.A1(new_n27), .A2(new_n52), .B(new_n24), .C(new_n72), .Y(new_n224));
  O2A1O1Ixp33_ASAP7_75t_R   g203(.A1(new_n52), .A2(new_n154), .B(new_n224), .C(new_n23), .Y(new_n225));
  INVx1_ASAP7_75t_R         g204(.A(new_n225), .Y(new_n226));
  O2A1O1Ixp33_ASAP7_75t_R   g205(.A1(new_n223), .A2(new_n28), .B(new_n226), .C(new_n31), .Y(new_n227));
  NAND2xp33_ASAP7_75t_R     g206(.A(new_n31), .B(new_n57), .Y(new_n228));
  O2A1O1Ixp33_ASAP7_75t_R   g207(.A1(new_n30), .A2(new_n223), .B(new_n228), .C(new_n52), .Y(new_n229));
  NOR2xp33_ASAP7_75t_R      g208(.A(new_n24), .B(\ctable[0] ), .Y(new_n230));
  NOR2xp33_ASAP7_75t_R      g209(.A(new_n64), .B(new_n183), .Y(new_n231));
  O2A1O1Ixp33_ASAP7_75t_R   g210(.A1(\ctable[1] ), .A2(new_n230), .B(new_n48), .C(new_n231), .Y(new_n232));
  AOI22xp33_ASAP7_75t_R     g211(.A1(\totalcoeffs[2] ), .A2(new_n203), .B1(\trailingones[1] ), .B2(new_n123), .Y(new_n233));
  OAI22xp33_ASAP7_75t_R     g212(.A1(\trailingones[0] ), .A2(new_n232), .B1(new_n22), .B2(new_n233), .Y(new_n234));
  INVx1_ASAP7_75t_R         g213(.A(new_n234), .Y(new_n235));
  OAI221xp5_ASAP7_75t_R     g214(.A1(new_n52), .A2(new_n31), .B1(\totalcoeffs[0] ), .B2(new_n172), .C(new_n122), .Y(new_n236));
  OAI31xp33_ASAP7_75t_R     g215(.A1(new_n24), .A2(new_n31), .A3(new_n63), .B(new_n236), .Y(new_n237));
  INVx1_ASAP7_75t_R         g216(.A(new_n237), .Y(new_n238));
  O2A1O1Ixp33_ASAP7_75t_R   g217(.A1(\trailingones[0] ), .A2(new_n135), .B(\totalcoeffs[2] ), .C(new_n30), .Y(new_n239));
  INVx1_ASAP7_75t_R         g218(.A(new_n137), .Y(new_n240));
  AOI211xp5_ASAP7_75t_R     g219(.A1(new_n23), .A2(new_n31), .B(new_n105), .C(new_n240), .Y(new_n241));
  NAND2xp33_ASAP7_75t_R     g220(.A(new_n27), .B(new_n203), .Y(new_n242));
  NAND2xp33_ASAP7_75t_R     g221(.A(\totalcoeffs[0] ), .B(new_n242), .Y(new_n243));
  AOI221xp5_ASAP7_75t_R     g222(.A1(new_n24), .A2(new_n137), .B1(new_n240), .B2(new_n243), .C(\ctable[0] ), .Y(new_n244));
  OAI31xp33_ASAP7_75t_R     g223(.A1(new_n239), .A2(new_n241), .A3(new_n244), .B(\totalcoeffs[3] ), .Y(new_n245));
  OAI221xp5_ASAP7_75t_R     g224(.A1(new_n27), .A2(new_n235), .B1(\totalcoeffs[1] ), .B2(new_n238), .C(new_n245), .Y(new_n246));
  O2A1O1Ixp33_ASAP7_75t_R   g225(.A1(new_n227), .A2(new_n229), .B(new_n25), .C(new_n246), .Y(new_n247));
  OAI31xp33_ASAP7_75t_R     g226(.A1(new_n23), .A2(new_n31), .A3(new_n154), .B(new_n203), .Y(new_n248));
  AOI21xp33_ASAP7_75t_R     g227(.A1(\ctable[2] ), .A2(new_n30), .B(new_n31), .Y(new_n249));
  OAI32xp33_ASAP7_75t_R     g228(.A1(\totalcoeffs[1] ), .A2(new_n24), .A3(new_n249), .B1(\totalcoeffs[2] ), .B2(new_n240), .Y(new_n250));
  AOI32xp33_ASAP7_75t_R     g229(.A1(new_n24), .A2(\ctable[2] ), .A3(new_n248), .B1(new_n23), .B2(new_n250), .Y(new_n251));
  OAI22xp33_ASAP7_75t_R     g230(.A1(\ctable[2] ), .A2(new_n247), .B1(new_n132), .B2(new_n251), .Y(new_n252));
  AOI22xp33_ASAP7_75t_R     g231(.A1(new_n29), .A2(new_n222), .B1(new_n70), .B2(new_n252), .Y(\ctoken_len[0] ));
  NOR2xp33_ASAP7_75t_R      g232(.A(\totalcoeffs[2] ), .B(new_n25), .Y(new_n254));
  NOR2xp33_ASAP7_75t_R      g233(.A(\ctable[1] ), .B(new_n101), .Y(new_n255));
  NOR2xp33_ASAP7_75t_R      g234(.A(new_n31), .B(new_n30), .Y(new_n256));
  INVx1_ASAP7_75t_R         g235(.A(new_n44), .Y(new_n257));
  AOI22xp33_ASAP7_75t_R     g236(.A1(new_n23), .A2(new_n93), .B1(\totalcoeffs[0] ), .B2(new_n82), .Y(new_n258));
  OAI332xp33_ASAP7_75t_R    g237(.A1(new_n24), .A2(new_n158), .A3(new_n255), .B1(new_n256), .B2(new_n195), .B3(new_n257), .C1(\trailingones[1] ), .C2(new_n258), .Y(new_n259));
  NOR2xp33_ASAP7_75t_R      g238(.A(\totalcoeffs[3] ), .B(new_n31), .Y(new_n260));
  OA33x2_ASAP7_75t_R        g239(.A1(\trailingones[1] ), .A2(new_n260), .A3(new_n254), .B1(new_n24), .B2(new_n101), .B3(new_n112), .Y(new_n261));
  AOI21xp33_ASAP7_75t_R     g240(.A1(\ctable[1] ), .A2(\trailingones[1] ), .B(new_n260), .Y(new_n262));
  OAI22xp33_ASAP7_75t_R     g241(.A1(new_n22), .A2(new_n145), .B1(new_n23), .B2(new_n262), .Y(new_n263));
  A2O1A1O1Ixp25_ASAP7_75t_R g242(.A1(new_n112), .A2(new_n194), .B(new_n172), .C(\trailingones[1] ), .D(new_n263), .Y(new_n264));
  OAI322xp33_ASAP7_75t_R    g243(.A1(new_n31), .A2(new_n30), .A3(new_n60), .B1(\ctable[1] ), .B2(new_n261), .C1(\totalcoeffs[2] ), .C2(new_n264), .Y(new_n265));
  AOI322xp5_ASAP7_75t_R     g244(.A1(new_n30), .A2(new_n172), .A3(new_n254), .B1(new_n27), .B2(new_n259), .C1(\totalcoeffs[1] ), .C2(new_n265), .Y(new_n266));
  NOR3xp33_ASAP7_75t_R      g245(.A(new_n52), .B(new_n30), .C(new_n31), .Y(new_n267));
  NOR2xp33_ASAP7_75t_R      g246(.A(new_n23), .B(new_n25), .Y(new_n268));
  NOR2xp33_ASAP7_75t_R      g247(.A(new_n24), .B(new_n25), .Y(new_n269));
  OAI221xp5_ASAP7_75t_R     g248(.A1(\totalcoeffs[2] ), .A2(new_n268), .B1(new_n147), .B2(new_n269), .C(\ctable[0] ), .Y(new_n270));
  OAI31xp33_ASAP7_75t_R     g249(.A1(new_n23), .A2(new_n25), .A3(new_n159), .B(new_n270), .Y(new_n271));
  NOR2xp33_ASAP7_75t_R      g250(.A(new_n25), .B(new_n52), .Y(new_n272));
  NOR2xp33_ASAP7_75t_R      g251(.A(new_n25), .B(new_n183), .Y(new_n273));
  O2A1O1Ixp33_ASAP7_75t_R   g252(.A1(new_n42), .A2(new_n272), .B(new_n23), .C(new_n273), .Y(new_n274));
  NOR2xp33_ASAP7_75t_R      g253(.A(new_n23), .B(new_n40), .Y(new_n275));
  NOR2xp33_ASAP7_75t_R      g254(.A(new_n39), .B(new_n92), .Y(new_n276));
  A2O1A1O1Ixp25_ASAP7_75t_R g255(.A1(new_n30), .A2(new_n104), .B(new_n275), .C(\trailingones[0] ), .D(new_n276), .Y(new_n277));
  NAND2xp33_ASAP7_75t_R     g256(.A(new_n122), .B(new_n272), .Y(new_n278));
  OAI221xp5_ASAP7_75t_R     g257(.A1(\trailingones[0] ), .A2(new_n274), .B1(\totalcoeffs[3] ), .B2(new_n277), .C(new_n278), .Y(new_n279));
  AOI322xp5_ASAP7_75t_R     g258(.A1(\totalcoeffs[2] ), .A2(\totalcoeffs[3] ), .A3(new_n267), .B1(new_n27), .B2(new_n271), .C1(\totalcoeffs[1] ), .C2(new_n279), .Y(new_n280));
  OAI22xp33_ASAP7_75t_R     g259(.A1(\ctable[0] ), .A2(new_n266), .B1(\ctable[1] ), .B2(new_n280), .Y(new_n281));
  NAND2xp33_ASAP7_75t_R     g260(.A(new_n25), .B(new_n52), .Y(new_n282));
  AOI32xp33_ASAP7_75t_R     g261(.A1(\ctable[2] ), .A2(new_n30), .A3(new_n108), .B1(new_n36), .B2(new_n146), .Y(new_n283));
  NOR3xp33_ASAP7_75t_R      g262(.A(new_n113), .B(new_n282), .C(new_n283), .Y(new_n284));
  NAND2xp33_ASAP7_75t_R     g263(.A(\totalcoeffs[4] ), .B(new_n29), .Y(new_n285));
  INVx1_ASAP7_75t_R         g264(.A(new_n58), .Y(new_n286));
  NOR5xp2_ASAP7_75t_R       g265(.A(\totalcoeffs[1] ), .B(\ctable[0] ), .C(\ctable[1] ), .D(new_n285), .E(new_n286), .Y(new_n287));
  A2O1A1O1Ixp25_ASAP7_75t_R g266(.A1(new_n29), .A2(new_n281), .B(new_n284), .C(new_n70), .D(new_n287), .Y(\ctoken_len[1] ));
  AOI211xp5_ASAP7_75t_R     g267(.A1(\totalcoeffs[1] ), .A2(\ctable[0] ), .B(\totalcoeffs[2] ), .C(new_n185), .Y(new_n289));
  OAI21xp33_ASAP7_75t_R     g268(.A1(\totalcoeffs[0] ), .A2(\totalcoeffs[3] ), .B(new_n289), .Y(new_n290));
  O2A1O1Ixp33_ASAP7_75t_R   g269(.A1(new_n37), .A2(new_n152), .B(new_n290), .C(new_n31), .Y(new_n291));
  AOI211xp5_ASAP7_75t_R     g270(.A1(new_n52), .A2(new_n144), .B(new_n91), .C(new_n272), .Y(new_n292));
  OAI221xp5_ASAP7_75t_R     g271(.A1(new_n25), .A2(new_n106), .B1(new_n230), .B2(new_n272), .C(\totalcoeffs[1] ), .Y(new_n293));
  OAI31xp33_ASAP7_75t_R     g272(.A1(new_n104), .A2(new_n95), .A3(new_n107), .B(new_n293), .Y(new_n294));
  INVx1_ASAP7_75t_R         g273(.A(new_n294), .Y(new_n295));
  INVx1_ASAP7_75t_R         g274(.A(new_n289), .Y(new_n296));
  AOI21xp33_ASAP7_75t_R     g275(.A1(new_n27), .A2(new_n52), .B(new_n155), .Y(new_n297));
  AOI32xp33_ASAP7_75t_R     g276(.A1(new_n24), .A2(\ctable[0] ), .A3(new_n57), .B1(\totalcoeffs[1] ), .B2(new_n171), .Y(new_n298));
  OA332x1_ASAP7_75t_R       g277(.A1(new_n57), .A2(new_n171), .A3(new_n296), .B1(new_n24), .B2(\totalcoeffs[3] ), .B3(new_n297), .C1(\totalcoeffs[0] ), .C2(new_n298), .Y(new_n299));
  OAI22xp33_ASAP7_75t_R     g278(.A1(new_n30), .A2(new_n295), .B1(\trailingones[0] ), .B2(new_n299), .Y(new_n300));
  O2A1O1Ixp33_ASAP7_75t_R   g279(.A1(new_n291), .A2(new_n292), .B(new_n30), .C(new_n300), .Y(new_n301));
  NOR2xp33_ASAP7_75t_R      g280(.A(new_n27), .B(\trailingones[0] ), .Y(new_n302));
  INVx1_ASAP7_75t_R         g281(.A(new_n243), .Y(new_n303));
  AOI221xp5_ASAP7_75t_R     g282(.A1(new_n27), .A2(\trailingones[1] ), .B1(new_n112), .B2(new_n240), .C(new_n22), .Y(new_n304));
  OAI321xp33_ASAP7_75t_R    g283(.A1(\ctable[1] ), .A2(new_n302), .A3(new_n303), .B1(\totalcoeffs[2] ), .B2(new_n304), .C(new_n185), .Y(new_n305));
  O2A1O1Ixp33_ASAP7_75t_R   g284(.A1(\ctable[1] ), .A2(new_n301), .B(new_n305), .C(\ctable[2] ), .Y(new_n306));
  AOI331xp33_ASAP7_75t_R    g285(.A1(new_n24), .A2(new_n31), .A3(new_n23), .B1(\totalcoeffs[2] ), .B2(\trailingones[0] ), .B3(new_n23), .C1(new_n108), .Y(new_n307));
  OAI32xp33_ASAP7_75t_R     g286(.A1(new_n30), .A2(new_n223), .A3(\trailingones[0] ), .B1(\trailingones[1] ), .B2(new_n307), .Y(new_n308));
  AOI32xp33_ASAP7_75t_R     g287(.A1(new_n23), .A2(new_n93), .A3(new_n153), .B1(new_n27), .B2(new_n308), .Y(new_n309));
  NOR2xp33_ASAP7_75t_R      g288(.A(new_n132), .B(new_n309), .Y(new_n310));
  NOR4xp25_ASAP7_75t_R      g289(.A(\totalcoeffs[2] ), .B(new_n133), .C(new_n282), .D(new_n285), .Y(new_n311));
  O2A1O1Ixp33_ASAP7_75t_R   g290(.A1(new_n306), .A2(new_n310), .B(new_n70), .C(new_n311), .Y(\ctoken_len[2] ));
  O2A1O1Ixp33_ASAP7_75t_R   g291(.A1(new_n22), .A2(new_n30), .B(\totalcoeffs[3] ), .C(new_n144), .Y(new_n313));
  O2A1O1Ixp33_ASAP7_75t_R   g292(.A1(\ctable[0] ), .A2(\trailingones[1] ), .B(new_n25), .C(new_n313), .Y(new_n314));
  OAI21xp33_ASAP7_75t_R     g293(.A1(\ctable[1] ), .A2(new_n267), .B(new_n25), .Y(new_n315));
  OAI221xp5_ASAP7_75t_R     g294(.A1(new_n209), .A2(new_n197), .B1(\totalcoeffs[2] ), .B2(new_n314), .C(new_n315), .Y(new_n316));
  OAI21xp33_ASAP7_75t_R     g295(.A1(\ctable[0] ), .A2(\ctable[1] ), .B(new_n213), .Y(new_n317));
  OAI221xp5_ASAP7_75t_R     g296(.A1(new_n22), .A2(new_n30), .B1(new_n144), .B2(new_n191), .C(\totalcoeffs[3] ), .Y(new_n318));
  AOI33xp33_ASAP7_75t_R     g297(.A1(new_n24), .A2(new_n317), .A3(new_n318), .B1(new_n59), .B2(new_n203), .B3(new_n209), .Y(new_n319));
  A2O1A1Ixp33_ASAP7_75t_R   g298(.A1(new_n105), .A2(new_n196), .B(\ctable[2] ), .C(new_n242), .Y(new_n320));
  OAI21xp33_ASAP7_75t_R     g299(.A1(\ctable[0] ), .A2(new_n59), .B(\ctable[1] ), .Y(new_n321));
  OAI221xp5_ASAP7_75t_R     g300(.A1(new_n29), .A2(new_n59), .B1(new_n70), .B2(new_n142), .C(new_n321), .Y(new_n322));
  O2A1O1Ixp33_ASAP7_75t_R   g301(.A1(\ctable[2] ), .A2(new_n142), .B(\totalcoeffs[0] ), .C(new_n322), .Y(new_n323));
  OAI211xp5_ASAP7_75t_R     g302(.A1(\totalcoeffs[1] ), .A2(new_n319), .B(new_n320), .C(new_n323), .Y(new_n324));
  AOI21xp33_ASAP7_75t_R     g303(.A1(\totalcoeffs[1] ), .A2(new_n316), .B(new_n324), .Y(\ctoken_len[3] ));
  O2A1O1Ixp33_ASAP7_75t_R   g304(.A1(new_n31), .A2(new_n30), .B(\totalcoeffs[1] ), .C(new_n303), .Y(new_n326));
  OAI31xp33_ASAP7_75t_R     g305(.A1(new_n24), .A2(new_n219), .A3(new_n326), .B(new_n220), .Y(new_n327));
  AND3x1_ASAP7_75t_R        g306(.A(new_n29), .B(new_n207), .C(new_n327), .Y(\ctoken_len[4] ));
endmodule


