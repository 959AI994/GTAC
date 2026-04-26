// Benchmark "../EPFL/benchmarks/random_control/cavlc" written by ABC on Tue Oct 28 13:10:14 2025

module \../EPFL/benchmarks/random_control/cavlc  ( 
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
    new_n77, new_n78, new_n79, new_n80, new_n81, new_n82, new_n83, new_n84,
    new_n85, new_n86, new_n87, new_n88, new_n89, new_n90, new_n91, new_n92,
    new_n93, new_n94, new_n95, new_n96, new_n97, new_n98, new_n99,
    new_n100, new_n101, new_n102, new_n103, new_n104, new_n105, new_n106,
    new_n107, new_n108, new_n109, new_n110, new_n111, new_n112, new_n113,
    new_n114, new_n115, new_n116, new_n117, new_n118, new_n119, new_n120,
    new_n121, new_n122, new_n123, new_n124, new_n125, new_n126, new_n127,
    new_n128, new_n129, new_n130, new_n131, new_n132, new_n133, new_n134,
    new_n135, new_n136, new_n137, new_n138, new_n139, new_n140, new_n141,
    new_n142, new_n144, new_n145, new_n146, new_n147, new_n148, new_n149,
    new_n150, new_n151, new_n152, new_n153, new_n154, new_n155, new_n156,
    new_n157, new_n158, new_n159, new_n160, new_n161, new_n162, new_n163,
    new_n164, new_n165, new_n166, new_n167, new_n168, new_n169, new_n170,
    new_n171, new_n172, new_n173, new_n174, new_n175, new_n176, new_n177,
    new_n178, new_n179, new_n180, new_n181, new_n182, new_n183, new_n184,
    new_n185, new_n186, new_n187, new_n188, new_n189, new_n190, new_n191,
    new_n192, new_n193, new_n194, new_n195, new_n196, new_n197, new_n198,
    new_n199, new_n200, new_n201, new_n202, new_n203, new_n204, new_n205,
    new_n206, new_n207, new_n208, new_n209, new_n210, new_n211, new_n212,
    new_n213, new_n214, new_n215, new_n216, new_n217, new_n218, new_n219,
    new_n220, new_n221, new_n222, new_n223, new_n224, new_n225, new_n226,
    new_n227, new_n228, new_n229, new_n230, new_n231, new_n232, new_n233,
    new_n234, new_n235, new_n236, new_n237, new_n238, new_n239, new_n240,
    new_n241, new_n242, new_n243, new_n244, new_n245, new_n246, new_n247,
    new_n248, new_n249, new_n250, new_n251, new_n252, new_n253, new_n255,
    new_n256, new_n257, new_n258, new_n259, new_n260, new_n261, new_n262,
    new_n263, new_n264, new_n265, new_n266, new_n267, new_n268, new_n269,
    new_n270, new_n271, new_n272, new_n273, new_n274, new_n275, new_n276,
    new_n277, new_n278, new_n279, new_n280, new_n281, new_n282, new_n283,
    new_n284, new_n285, new_n286, new_n287, new_n288, new_n289, new_n290,
    new_n291, new_n292, new_n293, new_n294, new_n295, new_n296, new_n297,
    new_n298, new_n299, new_n300, new_n301, new_n302, new_n303, new_n304,
    new_n305, new_n306, new_n307, new_n308, new_n309, new_n310, new_n311,
    new_n312, new_n313, new_n314, new_n315, new_n316, new_n317, new_n318,
    new_n319, new_n320, new_n321, new_n322, new_n323, new_n324, new_n325,
    new_n326, new_n327, new_n328, new_n329, new_n330, new_n331, new_n332,
    new_n333, new_n334, new_n335, new_n336, new_n337, new_n338, new_n339,
    new_n340, new_n341, new_n342, new_n343, new_n344, new_n345, new_n346,
    new_n347, new_n348, new_n349, new_n350, new_n352, new_n353, new_n354,
    new_n355, new_n356, new_n357, new_n358, new_n359, new_n360, new_n361,
    new_n362, new_n363, new_n364, new_n365, new_n366, new_n367, new_n368,
    new_n369, new_n370, new_n371, new_n372, new_n373, new_n374, new_n375,
    new_n376, new_n377, new_n378, new_n379, new_n380, new_n381, new_n382,
    new_n383, new_n384, new_n385, new_n386, new_n387, new_n388, new_n389,
    new_n390, new_n391, new_n392, new_n393, new_n394, new_n395, new_n396,
    new_n397, new_n398, new_n399, new_n400, new_n401, new_n402, new_n403,
    new_n404, new_n405, new_n406, new_n407, new_n408, new_n409, new_n410,
    new_n411, new_n412, new_n413, new_n414, new_n415, new_n416, new_n417,
    new_n418, new_n419, new_n420, new_n422, new_n423, new_n424, new_n425,
    new_n426, new_n427, new_n428, new_n430, new_n431, new_n432, new_n433,
    new_n435, new_n436, new_n437, new_n438, new_n439, new_n440, new_n441,
    new_n442, new_n443, new_n444, new_n445, new_n446, new_n447, new_n448,
    new_n449, new_n450, new_n451, new_n452, new_n453, new_n454, new_n455,
    new_n456, new_n457, new_n458, new_n459, new_n460, new_n461, new_n462,
    new_n463, new_n464, new_n465, new_n466, new_n467, new_n468, new_n469,
    new_n470, new_n471, new_n472, new_n473, new_n474, new_n475, new_n476,
    new_n477, new_n478, new_n479, new_n480, new_n481, new_n482, new_n483,
    new_n484, new_n485, new_n486, new_n487, new_n488, new_n489, new_n490,
    new_n491, new_n492, new_n493, new_n494, new_n495, new_n496, new_n497,
    new_n498, new_n499, new_n500, new_n501, new_n502, new_n503, new_n504,
    new_n505, new_n506, new_n507, new_n508, new_n509, new_n510, new_n511,
    new_n512, new_n513, new_n514, new_n515, new_n516, new_n517, new_n518,
    new_n520, new_n521, new_n522, new_n523, new_n524, new_n525, new_n526,
    new_n527, new_n528, new_n529, new_n530, new_n531, new_n532, new_n533,
    new_n534, new_n535, new_n536, new_n537, new_n538, new_n539, new_n540,
    new_n541, new_n542, new_n543, new_n544, new_n545, new_n546, new_n547,
    new_n548, new_n549, new_n550, new_n551, new_n552, new_n553, new_n554,
    new_n555, new_n556, new_n557, new_n558, new_n559, new_n560, new_n561,
    new_n562, new_n563, new_n564, new_n565, new_n566, new_n567, new_n568,
    new_n569, new_n570, new_n571, new_n572, new_n573, new_n574, new_n575,
    new_n576, new_n577, new_n578, new_n579, new_n580, new_n581, new_n582,
    new_n583, new_n584, new_n585, new_n586, new_n587, new_n588, new_n589,
    new_n590, new_n591, new_n592, new_n593, new_n594, new_n595, new_n596,
    new_n597, new_n598, new_n599, new_n600, new_n601, new_n602, new_n603,
    new_n604, new_n605, new_n607, new_n608, new_n609, new_n610, new_n611,
    new_n612, new_n613, new_n614, new_n615, new_n616, new_n617, new_n618,
    new_n619, new_n620, new_n621, new_n622, new_n623, new_n624, new_n625,
    new_n626, new_n627, new_n628, new_n629, new_n630, new_n631, new_n632,
    new_n633, new_n634, new_n635, new_n636, new_n637, new_n638, new_n639,
    new_n640, new_n641, new_n642, new_n643, new_n644, new_n645, new_n646,
    new_n647, new_n648, new_n649, new_n650, new_n651, new_n652, new_n653,
    new_n654, new_n655, new_n656, new_n657, new_n658, new_n659, new_n660,
    new_n661, new_n662, new_n663, new_n664, new_n665, new_n666, new_n667,
    new_n669, new_n670, new_n671, new_n672, new_n673, new_n674, new_n675,
    new_n676, new_n677, new_n678, new_n679, new_n680, new_n681, new_n682,
    new_n683, new_n684, new_n685, new_n686, new_n687, new_n688, new_n689,
    new_n690, new_n691, new_n692, new_n693, new_n694, new_n695, new_n696,
    new_n697, new_n698, new_n699, new_n700, new_n701, new_n702, new_n703,
    new_n704, new_n705, new_n706, new_n708, new_n709, new_n710, new_n711,
    new_n712, new_n713;
  assign new_n22 = ~\totalcoeffs[3]  & ~\ctable[0] ;
  assign new_n23 = ~\totalcoeffs[1]  & \trailingones[1] ;
  assign new_n24 = ~\totalcoeffs[0]  & \trailingones[1] ;
  assign new_n25 = ~new_n24 & \totalcoeffs[1] ;
  assign new_n26 = ~new_n23 & ~new_n25;
  assign new_n27 = new_n22 & new_n26;
  assign new_n28 = ~\trailingones[1]  & \totalcoeffs[0] ;
  assign new_n29 = ~\totalcoeffs[1]  & ~new_n28;
  assign new_n30 = ~\trailingones[1]  & \totalcoeffs[1] ;
  assign new_n31 = ~\ctable[2]  & ~new_n30;
  assign new_n32 = ~new_n29 & new_n31;
  assign new_n33 = ~new_n27 & ~new_n32;
  assign new_n34 = ~\totalcoeffs[2]  & ~new_n33;
  assign new_n35 = ~\totalcoeffs[2]  & ~\ctable[0] ;
  assign new_n36 = ~\totalcoeffs[0]  & ~new_n35;
  assign new_n37 = \totalcoeffs[1]  & \totalcoeffs[2] ;
  assign new_n38 = ~new_n36 & ~new_n37;
  assign new_n39 = ~\trailingones[1]  & ~new_n38;
  assign new_n40 = ~\totalcoeffs[2]  & ~\totalcoeffs[3] ;
  assign new_n41 = ~new_n40 & \trailingones[1] ;
  assign new_n42 = \totalcoeffs[0]  & \totalcoeffs[3] ;
  assign new_n43 = ~new_n41 & ~new_n42;
  assign new_n44 = ~\totalcoeffs[1]  & ~new_n43;
  assign new_n45 = ~\totalcoeffs[3]  & \totalcoeffs[1] ;
  assign new_n46 = ~new_n44 & ~new_n45;
  assign new_n47 = ~new_n39 & new_n46;
  assign new_n48 = ~\ctable[2]  & ~new_n47;
  assign new_n49 = ~new_n34 & ~new_n48;
  assign new_n50 = ~\trailingones[0]  & ~new_n49;
  assign new_n51 = \ctable[2]  & \trailingones[0] ;
  assign new_n52 = ~\ctable[0]  & new_n51;
  assign new_n53 = ~\ctable[2]  & \trailingones[1] ;
  assign new_n54 = ~new_n52 & ~new_n53;
  assign new_n55 = ~new_n54 & \totalcoeffs[1] ;
  assign new_n56 = ~\ctable[2]  & \totalcoeffs[1] ;
  assign new_n57 = ~\ctable[0]  & ~\trailingones[1] ;
  assign new_n58 = ~new_n56 & new_n57;
  assign new_n59 = ~new_n55 & ~new_n58;
  assign new_n60 = ~\totalcoeffs[2]  & \totalcoeffs[0] ;
  assign new_n61 = ~new_n59 & new_n60;
  assign new_n62 = ~\trailingones[1]  & \totalcoeffs[2] ;
  assign new_n63 = ~\totalcoeffs[0]  & ~\totalcoeffs[1] ;
  assign new_n64 = new_n62 & new_n63;
  assign new_n65 = new_n52 & new_n64;
  assign new_n66 = ~new_n61 & ~new_n65;
  assign new_n67 = ~\totalcoeffs[3]  & ~new_n66;
  assign new_n68 = ~new_n50 & ~new_n67;
  assign new_n69 = ~\ctable[1]  & ~new_n68;
  assign new_n70 = ~\totalcoeffs[0]  & ~\totalcoeffs[2] ;
  assign new_n71 = new_n23 & new_n70;
  assign new_n72 = ~\ctable[0]  & ~new_n71;
  assign new_n73 = ~new_n72 & \ctable[1] ;
  assign new_n74 = ~\totalcoeffs[0]  & ~\trailingones[1] ;
  assign new_n75 = \totalcoeffs[0]  & \trailingones[1] ;
  assign new_n76 = ~new_n74 & ~new_n75;
  assign new_n77 = ~new_n76 & \totalcoeffs[1] ;
  assign new_n78 = \totalcoeffs[2]  & new_n77;
  assign new_n79 = \ctable[0]  & new_n78;
  assign new_n80 = ~new_n73 & ~new_n79;
  assign new_n81 = ~new_n80 & \trailingones[0] ;
  assign new_n82 = ~\totalcoeffs[1]  & ~new_n24;
  assign new_n83 = ~\trailingones[0]  & ~new_n82;
  assign new_n84 = ~\totalcoeffs[1]  & \totalcoeffs[2] ;
  assign new_n85 = new_n28 & new_n84;
  assign new_n86 = ~new_n83 & ~new_n85;
  assign new_n87 = ~\ctable[0]  & ~new_n86;
  assign new_n88 = ~new_n81 & ~new_n87;
  assign new_n89 = ~new_n88 & \totalcoeffs[3] ;
  assign new_n90 = ~\totalcoeffs[2]  & ~new_n28;
  assign new_n91 = ~new_n90 & \ctable[1] ;
  assign new_n92 = new_n40 & new_n77;
  assign new_n93 = ~new_n91 & ~new_n92;
  assign new_n94 = ~new_n93 & \ctable[0] ;
  assign new_n95 = \totalcoeffs[2]  & \trailingones[1] ;
  assign new_n96 = ~\totalcoeffs[2]  & new_n30;
  assign new_n97 = ~new_n95 & ~new_n96;
  assign new_n98 = ~\totalcoeffs[0]  & ~\totalcoeffs[3] ;
  assign new_n99 = ~new_n97 & new_n98;
  assign new_n100 = \ctable[1]  & new_n99;
  assign new_n101 = ~new_n94 & ~new_n100;
  assign new_n102 = ~new_n101 & \trailingones[0] ;
  assign new_n103 = ~\totalcoeffs[3]  & \totalcoeffs[2] ;
  assign new_n104 = new_n23 & new_n103;
  assign new_n105 = ~\trailingones[0]  & \totalcoeffs[1] ;
  assign new_n106 = ~new_n104 & ~new_n105;
  assign new_n107 = ~\totalcoeffs[0]  & ~new_n106;
  assign new_n108 = ~\totalcoeffs[2]  & \ctable[1] ;
  assign new_n109 = ~new_n108 & \totalcoeffs[3] ;
  assign new_n110 = ~\totalcoeffs[1]  & ~\trailingones[1] ;
  assign new_n111 = ~new_n109 & new_n110;
  assign new_n112 = ~new_n95 & ~new_n111;
  assign new_n113 = ~\trailingones[0]  & ~new_n112;
  assign new_n114 = ~new_n107 & ~new_n113;
  assign new_n115 = ~\ctable[0]  & ~new_n114;
  assign new_n116 = ~\totalcoeffs[0]  & new_n40;
  assign new_n117 = ~\trailingones[0]  & new_n110;
  assign new_n118 = new_n116 & new_n117;
  assign new_n119 = ~new_n115 & ~new_n118;
  assign new_n120 = ~new_n102 & new_n119;
  assign new_n121 = ~new_n89 & new_n120;
  assign new_n122 = ~\ctable[2]  & ~new_n121;
  assign new_n123 = ~new_n69 & ~new_n122;
  assign new_n124 = ~\totalcoeffs[4]  & ~new_n123;
  assign new_n125 = ~\trailingones[0]  & ~\trailingones[1] ;
  assign new_n126 = ~\ctable[1]  & new_n125;
  assign new_n127 = \totalcoeffs[4]  & \trailingones[0] ;
  assign new_n128 = \ctable[1]  & new_n127;
  assign new_n129 = ~new_n126 & ~new_n128;
  assign new_n130 = ~new_n129 & \ctable[0] ;
  assign new_n131 = ~\trailingones[1]  & \ctable[1] ;
  assign new_n132 = \totalcoeffs[4]  & \trailingones[1] ;
  assign new_n133 = ~new_n131 & ~new_n132;
  assign new_n134 = \ctable[0]  & \ctable[1] ;
  assign new_n135 = ~new_n133 & ~new_n134;
  assign new_n136 = ~\trailingones[0]  & new_n135;
  assign new_n137 = ~new_n130 & ~new_n136;
  assign new_n138 = ~\ctable[2]  & ~new_n137;
  assign new_n139 = ~\totalcoeffs[1]  & ~\totalcoeffs[3] ;
  assign new_n140 = ~\totalcoeffs[2]  & new_n139;
  assign new_n141 = new_n138 & new_n140;
  assign new_n142 = ~\totalcoeffs[0]  & new_n141;
  assign \coeff_token[0]  = new_n124 | new_n142;
  assign new_n144 = \totalcoeffs[0]  & \trailingones[0] ;
  assign new_n145 = new_n95 & new_n144;
  assign new_n146 = ~\totalcoeffs[2]  & ~\trailingones[0] ;
  assign new_n147 = new_n74 & new_n146;
  assign new_n148 = ~new_n145 & ~new_n147;
  assign new_n149 = ~new_n148 & \ctable[0] ;
  assign new_n150 = ~\ctable[0]  & \trailingones[0] ;
  assign new_n151 = \ctable[1]  & \trailingones[1] ;
  assign new_n152 = ~new_n150 & ~new_n151;
  assign new_n153 = ~\totalcoeffs[0]  & ~new_n152;
  assign new_n154 = \trailingones[0]  & new_n151;
  assign new_n155 = ~\trailingones[0]  & new_n57;
  assign new_n156 = ~new_n154 & ~new_n155;
  assign new_n157 = ~new_n153 & new_n156;
  assign new_n158 = ~new_n157 & \totalcoeffs[2] ;
  assign new_n159 = ~new_n149 & ~new_n158;
  assign new_n160 = ~\totalcoeffs[4]  & ~new_n159;
  assign new_n161 = ~\ctable[1]  & \trailingones[1] ;
  assign new_n162 = ~\trailingones[0]  & \ctable[0] ;
  assign new_n163 = ~new_n127 & ~new_n162;
  assign new_n164 = ~\ctable[1]  & ~new_n163;
  assign new_n165 = ~new_n132 & ~new_n164;
  assign new_n166 = ~new_n161 & ~new_n165;
  assign new_n167 = new_n70 & new_n166;
  assign new_n168 = ~new_n160 & ~new_n167;
  assign new_n169 = ~\totalcoeffs[1]  & ~new_n168;
  assign new_n170 = \ctable[1]  & \trailingones[0] ;
  assign new_n171 = ~new_n152 & ~new_n170;
  assign new_n172 = \totalcoeffs[0]  & new_n171;
  assign new_n173 = ~\totalcoeffs[0]  & \ctable[0] ;
  assign new_n174 = ~\trailingones[0]  & \trailingones[1] ;
  assign new_n175 = new_n173 & new_n174;
  assign new_n176 = ~new_n172 & ~new_n175;
  assign new_n177 = ~\totalcoeffs[2]  & ~new_n176;
  assign new_n178 = ~\ctable[1]  & ~\trailingones[0] ;
  assign new_n179 = ~\totalcoeffs[0]  & \ctable[1] ;
  assign new_n180 = ~\totalcoeffs[2]  & ~new_n179;
  assign new_n181 = ~new_n180 & new_n150;
  assign new_n182 = ~new_n178 & ~new_n181;
  assign new_n183 = ~\trailingones[1]  & ~new_n182;
  assign new_n184 = ~new_n177 & ~new_n183;
  assign new_n185 = ~new_n184 & \totalcoeffs[1] ;
  assign new_n186 = ~\ctable[1]  & \trailingones[0] ;
  assign new_n187 = new_n62 & new_n186;
  assign new_n188 = ~new_n185 & ~new_n187;
  assign new_n189 = ~\totalcoeffs[4]  & ~new_n188;
  assign new_n190 = ~new_n169 & ~new_n189;
  assign new_n191 = ~\totalcoeffs[3]  & ~new_n190;
  assign new_n192 = \ctable[0]  & new_n146;
  assign new_n193 = ~\totalcoeffs[3]  & ~\trailingones[0] ;
  assign new_n194 = ~new_n150 & ~new_n193;
  assign new_n195 = \totalcoeffs[0]  & new_n194;
  assign new_n196 = ~new_n192 & ~new_n195;
  assign new_n197 = ~\totalcoeffs[1]  & ~new_n196;
  assign new_n198 = \totalcoeffs[3]  & \trailingones[0] ;
  assign new_n199 = ~\totalcoeffs[2]  & ~new_n198;
  assign new_n200 = ~\totalcoeffs[0]  & ~new_n199;
  assign new_n201 = ~\totalcoeffs[2]  & \trailingones[0] ;
  assign new_n202 = \totalcoeffs[0]  & new_n201;
  assign new_n203 = ~\ctable[0]  & \totalcoeffs[0] ;
  assign new_n204 = ~new_n173 & ~new_n203;
  assign new_n205 = ~new_n202 & new_n204;
  assign new_n206 = ~new_n205 & \totalcoeffs[1] ;
  assign new_n207 = ~new_n200 & ~new_n206;
  assign new_n208 = ~new_n197 & new_n207;
  assign new_n209 = ~\ctable[1]  & ~new_n208;
  assign new_n210 = ~\totalcoeffs[1]  & \ctable[1] ;
  assign new_n211 = ~new_n210 & \totalcoeffs[2] ;
  assign new_n212 = ~\totalcoeffs[1]  & ~\trailingones[0] ;
  assign new_n213 = ~new_n211 & ~new_n212;
  assign new_n214 = \totalcoeffs[3]  & new_n213;
  assign new_n215 = ~\totalcoeffs[0]  & \trailingones[0] ;
  assign new_n216 = ~\totalcoeffs[1]  & ~\ctable[1] ;
  assign new_n217 = \totalcoeffs[0]  & \totalcoeffs[1] ;
  assign new_n218 = ~new_n216 & ~new_n217;
  assign new_n219 = ~new_n215 & new_n218;
  assign new_n220 = ~\totalcoeffs[2]  & new_n219;
  assign new_n221 = ~new_n214 & ~new_n220;
  assign new_n222 = ~\ctable[0]  & ~new_n221;
  assign new_n223 = ~new_n209 & ~new_n222;
  assign new_n224 = ~\trailingones[1]  & ~new_n223;
  assign new_n225 = \totalcoeffs[1]  & \ctable[0] ;
  assign new_n226 = ~\trailingones[0]  & \totalcoeffs[3] ;
  assign new_n227 = ~\totalcoeffs[1]  & ~\totalcoeffs[2] ;
  assign new_n228 = new_n226 & new_n227;
  assign new_n229 = ~new_n225 & ~new_n228;
  assign new_n230 = ~new_n229 & \totalcoeffs[0] ;
  assign new_n231 = ~new_n40 & \ctable[0] ;
  assign new_n232 = \totalcoeffs[3]  & new_n37;
  assign new_n233 = ~new_n231 & ~new_n232;
  assign new_n234 = ~new_n230 & new_n233;
  assign new_n235 = ~new_n234 & \ctable[1] ;
  assign new_n236 = new_n162 & new_n232;
  assign new_n237 = ~new_n235 & ~new_n236;
  assign new_n238 = ~new_n237 & \trailingones[1] ;
  assign new_n239 = ~new_n224 & ~new_n238;
  assign new_n240 = ~\totalcoeffs[4]  & ~new_n239;
  assign new_n241 = ~new_n191 & ~new_n240;
  assign new_n242 = ~\ctable[2]  & ~new_n241;
  assign new_n243 = ~new_n63 & \totalcoeffs[2] ;
  assign new_n244 = ~new_n217 & new_n90;
  assign new_n245 = ~new_n243 & ~new_n244;
  assign new_n246 = ~\trailingones[0]  & new_n245;
  assign new_n247 = new_n30 & new_n201;
  assign new_n248 = ~new_n246 & ~new_n247;
  assign new_n249 = ~new_n248 & \ctable[2] ;
  assign new_n250 = ~new_n64 & ~new_n249;
  assign new_n251 = ~\totalcoeffs[4]  & ~new_n250;
  assign new_n252 = ~\ctable[1]  & new_n251;
  assign new_n253 = new_n22 & new_n252;
  assign \coeff_token[1]  = new_n242 | new_n253;
  assign new_n255 = ~\totalcoeffs[3]  & \trailingones[1] ;
  assign new_n256 = ~new_n45 & ~new_n174;
  assign new_n257 = ~new_n255 & ~new_n256;
  assign new_n258 = ~\ctable[1]  & new_n257;
  assign new_n259 = ~new_n201 & \totalcoeffs[3] ;
  assign new_n260 = ~new_n259 & \ctable[0] ;
  assign new_n261 = ~\ctable[0]  & \totalcoeffs[2] ;
  assign new_n262 = ~new_n261 & \totalcoeffs[3] ;
  assign new_n263 = ~new_n262 & \trailingones[0] ;
  assign new_n264 = ~new_n260 & ~new_n263;
  assign new_n265 = ~new_n264 & \totalcoeffs[1] ;
  assign new_n266 = ~\ctable[0]  & \ctable[1] ;
  assign new_n267 = ~\trailingones[0]  & new_n266;
  assign new_n268 = new_n139 & new_n267;
  assign new_n269 = ~new_n265 & ~new_n268;
  assign new_n270 = ~\trailingones[1]  & ~new_n269;
  assign new_n271 = ~\totalcoeffs[1]  & ~new_n199;
  assign new_n272 = ~new_n186 & \totalcoeffs[2] ;
  assign new_n273 = ~new_n271 & ~new_n272;
  assign new_n274 = ~new_n273 & \ctable[0] ;
  assign new_n275 = \totalcoeffs[1]  & \trailingones[0] ;
  assign new_n276 = ~new_n275 & \ctable[1] ;
  assign new_n277 = ~\trailingones[1]  & ~new_n276;
  assign new_n278 = ~\totalcoeffs[2]  & ~new_n277;
  assign new_n279 = ~\totalcoeffs[1]  & new_n174;
  assign new_n280 = ~new_n278 & ~new_n279;
  assign new_n281 = ~new_n280 & \totalcoeffs[3] ;
  assign new_n282 = ~\trailingones[0]  & \ctable[1] ;
  assign new_n283 = \totalcoeffs[1]  & new_n255;
  assign new_n284 = new_n282 & new_n283;
  assign new_n285 = ~new_n281 & ~new_n284;
  assign new_n286 = ~new_n274 & new_n285;
  assign new_n287 = ~new_n270 & new_n286;
  assign new_n288 = ~new_n258 & new_n287;
  assign new_n289 = ~\totalcoeffs[4]  & ~new_n288;
  assign new_n290 = ~\trailingones[1]  & \trailingones[0] ;
  assign new_n291 = ~new_n178 & ~new_n290;
  assign new_n292 = ~\ctable[0]  & new_n291;
  assign new_n293 = ~\totalcoeffs[3]  & \totalcoeffs[4] ;
  assign new_n294 = ~\totalcoeffs[1]  & new_n293;
  assign new_n295 = ~new_n292 & new_n294;
  assign new_n296 = ~\totalcoeffs[2]  & new_n295;
  assign new_n297 = ~new_n289 & ~new_n296;
  assign new_n298 = ~\totalcoeffs[0]  & ~new_n297;
  assign new_n299 = \trailingones[0]  & new_n75;
  assign new_n300 = ~new_n125 & ~new_n299;
  assign new_n301 = ~new_n300 & \ctable[0] ;
  assign new_n302 = ~\ctable[0]  & new_n95;
  assign new_n303 = ~new_n301 & ~new_n302;
  assign new_n304 = ~new_n303 & \totalcoeffs[1] ;
  assign new_n305 = ~new_n174 & ~new_n290;
  assign new_n306 = ~new_n110 & new_n305;
  assign new_n307 = ~new_n306 & \totalcoeffs[2] ;
  assign new_n308 = ~new_n304 & ~new_n307;
  assign new_n309 = ~\totalcoeffs[3]  & ~new_n308;
  assign new_n310 = ~\totalcoeffs[1]  & \totalcoeffs[3] ;
  assign new_n311 = ~new_n310 & \ctable[0] ;
  assign new_n312 = ~\totalcoeffs[2]  & new_n125;
  assign new_n313 = ~new_n311 & new_n312;
  assign new_n314 = ~\totalcoeffs[1]  & \trailingones[0] ;
  assign new_n315 = new_n41 & new_n314;
  assign new_n316 = ~new_n313 & ~new_n315;
  assign new_n317 = ~new_n316 & \totalcoeffs[0] ;
  assign new_n318 = ~new_n309 & ~new_n317;
  assign new_n319 = ~\ctable[1]  & ~new_n318;
  assign new_n320 = \totalcoeffs[1]  & \trailingones[1] ;
  assign new_n321 = \totalcoeffs[2]  & new_n320;
  assign new_n322 = ~new_n314 & ~new_n321;
  assign new_n323 = ~new_n322 & \ctable[1] ;
  assign new_n324 = ~new_n62 & \trailingones[0] ;
  assign new_n325 = \totalcoeffs[2]  & new_n125;
  assign new_n326 = ~new_n324 & ~new_n325;
  assign new_n327 = ~new_n323 & new_n326;
  assign new_n328 = ~new_n327 & \totalcoeffs[3] ;
  assign new_n329 = ~new_n23 & \ctable[1] ;
  assign new_n330 = ~new_n30 & ~new_n329;
  assign new_n331 = ~new_n330 & \trailingones[0] ;
  assign new_n332 = ~new_n110 & ~new_n320;
  assign new_n333 = ~new_n332 & new_n193;
  assign new_n334 = ~new_n331 & ~new_n333;
  assign new_n335 = ~\totalcoeffs[2]  & ~new_n334;
  assign new_n336 = ~new_n328 & ~new_n335;
  assign new_n337 = ~new_n336 & new_n203;
  assign new_n338 = ~new_n319 & ~new_n337;
  assign new_n339 = ~\totalcoeffs[4]  & ~new_n338;
  assign new_n340 = ~new_n298 & ~new_n339;
  assign new_n341 = ~\ctable[2]  & ~new_n340;
  assign new_n342 = ~\ctable[0]  & ~\ctable[1] ;
  assign new_n343 = new_n51 & new_n75;
  assign new_n344 = ~new_n74 & ~new_n343;
  assign new_n345 = ~new_n344 & \totalcoeffs[1] ;
  assign new_n346 = new_n28 & new_n212;
  assign new_n347 = ~new_n345 & ~new_n346;
  assign new_n348 = ~\totalcoeffs[4]  & ~new_n347;
  assign new_n349 = new_n40 & new_n348;
  assign new_n350 = new_n342 & new_n349;
  assign \coeff_token[2]  = new_n341 | new_n350;
  assign new_n352 = ~new_n161 & \totalcoeffs[3] ;
  assign new_n353 = \totalcoeffs[2]  & \ctable[1] ;
  assign new_n354 = ~new_n352 & ~new_n353;
  assign new_n355 = ~\totalcoeffs[4]  & ~new_n354;
  assign new_n356 = \trailingones[0]  & \trailingones[1] ;
  assign new_n357 = new_n342 & new_n356;
  assign new_n358 = ~new_n134 & ~new_n357;
  assign new_n359 = ~\totalcoeffs[2]  & new_n293;
  assign new_n360 = ~new_n358 & new_n359;
  assign new_n361 = ~new_n355 & ~new_n360;
  assign new_n362 = ~\totalcoeffs[1]  & ~new_n361;
  assign new_n363 = \ctable[0]  & \trailingones[0] ;
  assign new_n364 = new_n161 & new_n363;
  assign new_n365 = ~new_n155 & ~new_n364;
  assign new_n366 = ~new_n365 & \totalcoeffs[2] ;
  assign new_n367 = ~new_n186 & \ctable[0] ;
  assign new_n368 = ~new_n367 & \totalcoeffs[3] ;
  assign new_n369 = ~new_n267 & ~new_n368;
  assign new_n370 = ~new_n366 & new_n369;
  assign new_n371 = ~new_n370 & \totalcoeffs[1] ;
  assign new_n372 = \totalcoeffs[2]  & new_n266;
  assign new_n373 = ~new_n371 & ~new_n372;
  assign new_n374 = ~\totalcoeffs[4]  & ~new_n373;
  assign new_n375 = ~new_n362 & ~new_n374;
  assign new_n376 = ~\totalcoeffs[0]  & ~new_n375;
  assign new_n377 = ~new_n194 & \totalcoeffs[2] ;
  assign new_n378 = ~\totalcoeffs[2]  & new_n363;
  assign new_n379 = ~new_n377 & ~new_n378;
  assign new_n380 = ~\trailingones[1]  & ~new_n379;
  assign new_n381 = \ctable[0]  & \trailingones[1] ;
  assign new_n382 = ~\trailingones[0]  & new_n381;
  assign new_n383 = ~\totalcoeffs[3]  & ~new_n382;
  assign new_n384 = ~\totalcoeffs[2]  & ~new_n383;
  assign new_n385 = ~\ctable[1]  & ~new_n226;
  assign new_n386 = ~\ctable[0]  & \totalcoeffs[3] ;
  assign new_n387 = ~new_n385 & ~new_n386;
  assign new_n388 = ~new_n384 & ~new_n387;
  assign new_n389 = ~new_n380 & new_n388;
  assign new_n390 = ~new_n389 & \totalcoeffs[1] ;
  assign new_n391 = ~\totalcoeffs[3]  & new_n266;
  assign new_n392 = new_n192 & new_n216;
  assign new_n393 = ~new_n391 & ~new_n392;
  assign new_n394 = ~\trailingones[1]  & ~new_n393;
  assign new_n395 = ~new_n390 & ~new_n394;
  assign new_n396 = ~new_n395 & \totalcoeffs[0] ;
  assign new_n397 = ~new_n40 & new_n266;
  assign new_n398 = ~\ctable[1]  & \totalcoeffs[3] ;
  assign new_n399 = \totalcoeffs[2]  & new_n398;
  assign new_n400 = ~new_n397 & ~new_n399;
  assign new_n401 = ~new_n400 & \trailingones[1] ;
  assign new_n402 = ~\trailingones[1]  & new_n266;
  assign new_n403 = ~new_n398 & ~new_n402;
  assign new_n404 = ~\trailingones[0]  & ~new_n403;
  assign new_n405 = ~new_n401 & ~new_n404;
  assign new_n406 = ~\totalcoeffs[1]  & ~new_n405;
  assign new_n407 = ~new_n57 & ~new_n178;
  assign new_n408 = ~\totalcoeffs[2]  & ~new_n407;
  assign new_n409 = ~new_n320 & ~new_n363;
  assign new_n410 = ~\ctable[1]  & ~new_n381;
  assign new_n411 = ~new_n409 & new_n410;
  assign new_n412 = ~new_n408 & ~new_n411;
  assign new_n413 = ~new_n412 & \totalcoeffs[3] ;
  assign new_n414 = \totalcoeffs[1]  & new_n290;
  assign new_n415 = new_n266 & new_n414;
  assign new_n416 = ~new_n413 & ~new_n415;
  assign new_n417 = ~new_n406 & new_n416;
  assign new_n418 = ~new_n396 & new_n417;
  assign new_n419 = ~\totalcoeffs[4]  & ~new_n418;
  assign new_n420 = ~new_n376 & ~new_n419;
  assign \coeff_token[3]  = ~\ctable[2]  & ~new_n420;
  assign new_n422 = ~\totalcoeffs[4]  & new_n243;
  assign new_n423 = ~\totalcoeffs[4]  & \totalcoeffs[3] ;
  assign new_n424 = ~new_n293 & ~new_n423;
  assign new_n425 = ~\totalcoeffs[0]  & new_n227;
  assign new_n426 = ~new_n424 & new_n425;
  assign new_n427 = ~new_n422 & ~new_n426;
  assign new_n428 = ~\ctable[2]  & ~new_n427;
  assign \coeff_token[4]  = new_n134 & new_n428;
  assign new_n430 = new_n63 & new_n359;
  assign new_n431 = ~new_n425 & new_n423;
  assign new_n432 = ~new_n430 & ~new_n431;
  assign new_n433 = ~\ctable[2]  & ~new_n432;
  assign \coeff_token[5]  = new_n134 & new_n433;
  assign new_n435 = ~new_n125 & ~new_n356;
  assign new_n436 = ~new_n435 & \ctable[1] ;
  assign new_n437 = ~\totalcoeffs[2]  & \trailingones[1] ;
  assign new_n438 = ~new_n62 & ~new_n437;
  assign new_n439 = new_n436 & new_n438;
  assign new_n440 = ~\totalcoeffs[1]  & ~\ctable[0] ;
  assign new_n441 = ~new_n440 & \totalcoeffs[2] ;
  assign new_n442 = ~\trailingones[1]  & ~new_n441;
  assign new_n443 = ~\ctable[1]  & new_n442;
  assign new_n444 = \ctable[0]  & new_n320;
  assign new_n445 = ~new_n443 & ~new_n444;
  assign new_n446 = ~new_n445 & \totalcoeffs[0] ;
  assign new_n447 = ~\totalcoeffs[0]  & ~new_n57;
  assign new_n448 = \totalcoeffs[2]  & new_n447;
  assign new_n449 = ~new_n446 & ~new_n448;
  assign new_n450 = ~new_n449 & \trailingones[0] ;
  assign new_n451 = \totalcoeffs[2]  & new_n24;
  assign new_n452 = ~new_n117 & ~new_n451;
  assign new_n453 = ~new_n452 & \ctable[0] ;
  assign new_n454 = ~new_n450 & ~new_n453;
  assign new_n455 = ~new_n439 & new_n454;
  assign new_n456 = ~\ctable[2]  & ~new_n455;
  assign new_n457 = ~\trailingones[1]  & \ctable[2] ;
  assign new_n458 = ~new_n457 & \trailingones[0] ;
  assign new_n459 = ~new_n458 & new_n84;
  assign new_n460 = ~new_n96 & ~new_n459;
  assign new_n461 = ~\totalcoeffs[0]  & ~new_n460;
  assign new_n462 = new_n144 & new_n320;
  assign new_n463 = ~new_n125 & ~new_n462;
  assign new_n464 = ~new_n463 & \ctable[2] ;
  assign new_n465 = ~\totalcoeffs[2]  & new_n464;
  assign new_n466 = ~new_n461 & ~new_n465;
  assign new_n467 = ~new_n466 & new_n342;
  assign new_n468 = ~new_n456 & ~new_n467;
  assign new_n469 = ~\totalcoeffs[3]  & ~new_n468;
  assign new_n470 = ~\totalcoeffs[0]  & ~\ctable[1] ;
  assign new_n471 = ~new_n212 & new_n332;
  assign new_n472 = new_n470 & new_n471;
  assign new_n473 = ~new_n117 & ~new_n320;
  assign new_n474 = ~new_n473 & \totalcoeffs[0] ;
  assign new_n475 = \totalcoeffs[2]  & new_n30;
  assign new_n476 = ~new_n474 & ~new_n475;
  assign new_n477 = ~new_n472 & new_n476;
  assign new_n478 = ~\ctable[0]  & ~new_n477;
  assign new_n479 = ~\trailingones[0]  & new_n217;
  assign new_n480 = \totalcoeffs[0]  & new_n170;
  assign new_n481 = ~new_n105 & ~new_n480;
  assign new_n482 = ~\totalcoeffs[2]  & ~new_n481;
  assign new_n483 = ~new_n479 & ~new_n482;
  assign new_n484 = ~new_n483 & \trailingones[1] ;
  assign new_n485 = ~\totalcoeffs[0]  & ~\trailingones[0] ;
  assign new_n486 = ~new_n203 & ~new_n485;
  assign new_n487 = new_n30 & new_n486;
  assign new_n488 = ~new_n484 & ~new_n487;
  assign new_n489 = ~new_n478 & new_n488;
  assign new_n490 = ~new_n489 & \totalcoeffs[3] ;
  assign new_n491 = ~\totalcoeffs[2]  & \ctable[0] ;
  assign new_n492 = ~new_n76 & new_n491;
  assign new_n493 = ~\ctable[1]  & ~new_n261;
  assign new_n494 = ~new_n493 & new_n24;
  assign new_n495 = ~new_n492 & ~new_n494;
  assign new_n496 = ~\trailingones[0]  & ~new_n495;
  assign new_n497 = ~new_n125 & \totalcoeffs[2] ;
  assign new_n498 = ~new_n299 & ~new_n497;
  assign new_n499 = ~new_n498 & \ctable[1] ;
  assign new_n500 = ~new_n496 & ~new_n499;
  assign new_n501 = ~new_n500 & \totalcoeffs[1] ;
  assign new_n502 = ~\trailingones[0]  & ~new_n470;
  assign new_n503 = ~new_n203 & ~new_n502;
  assign new_n504 = ~\trailingones[1]  & ~new_n503;
  assign new_n505 = ~\totalcoeffs[2]  & new_n504;
  assign new_n506 = \totalcoeffs[2]  & \trailingones[0] ;
  assign new_n507 = new_n24 & new_n506;
  assign new_n508 = ~new_n505 & ~new_n507;
  assign new_n509 = ~\totalcoeffs[1]  & ~new_n508;
  assign new_n510 = ~new_n501 & ~new_n509;
  assign new_n511 = ~new_n490 & new_n510;
  assign new_n512 = ~\ctable[2]  & ~new_n511;
  assign new_n513 = ~new_n469 & ~new_n512;
  assign new_n514 = ~\totalcoeffs[4]  & ~new_n513;
  assign new_n515 = ~\totalcoeffs[1]  & \totalcoeffs[4] ;
  assign new_n516 = new_n116 & new_n515;
  assign new_n517 = ~new_n134 & ~new_n516;
  assign new_n518 = ~\ctable[2]  & ~new_n517;
  assign \ctoken_len[0]  = ~new_n514 & ~new_n518;
  assign new_n520 = ~\totalcoeffs[3]  & \trailingones[0] ;
  assign new_n521 = ~new_n151 & ~new_n520;
  assign new_n522 = ~new_n521 & \totalcoeffs[0] ;
  assign new_n523 = new_n215 & new_n398;
  assign new_n524 = ~new_n282 & ~new_n523;
  assign new_n525 = ~new_n524 & \trailingones[1] ;
  assign new_n526 = ~\trailingones[1]  & new_n170;
  assign new_n527 = ~new_n525 & ~new_n526;
  assign new_n528 = ~new_n522 & new_n527;
  assign new_n529 = ~\totalcoeffs[2]  & ~new_n528;
  assign new_n530 = ~\totalcoeffs[3]  & new_n356;
  assign new_n531 = ~new_n356 & \totalcoeffs[3] ;
  assign new_n532 = ~new_n144 & ~new_n531;
  assign new_n533 = ~\ctable[1]  & ~new_n532;
  assign new_n534 = ~new_n530 & ~new_n533;
  assign new_n535 = ~new_n534 & \totalcoeffs[2] ;
  assign new_n536 = ~\totalcoeffs[3]  & new_n126;
  assign new_n537 = ~new_n535 & ~new_n536;
  assign new_n538 = ~new_n529 & new_n537;
  assign new_n539 = ~new_n538 & \totalcoeffs[1] ;
  assign new_n540 = ~new_n74 & ~new_n174;
  assign new_n541 = ~\ctable[1]  & ~new_n540;
  assign new_n542 = ~\totalcoeffs[2]  & new_n541;
  assign new_n543 = ~\totalcoeffs[0]  & new_n436;
  assign new_n544 = ~new_n542 & ~new_n543;
  assign new_n545 = ~new_n544 & \totalcoeffs[3] ;
  assign new_n546 = ~\totalcoeffs[0]  & new_n146;
  assign new_n547 = ~new_n480 & ~new_n546;
  assign new_n548 = ~\trailingones[1]  & ~new_n547;
  assign new_n549 = ~\ctable[1]  & ~new_n193;
  assign new_n550 = ~new_n549 & new_n497;
  assign new_n551 = ~new_n548 & ~new_n550;
  assign new_n552 = ~new_n545 & new_n551;
  assign new_n553 = ~\totalcoeffs[1]  & ~new_n552;
  assign new_n554 = ~\totalcoeffs[2]  & \totalcoeffs[3] ;
  assign new_n555 = ~\trailingones[1]  & new_n282;
  assign new_n556 = new_n554 & new_n555;
  assign new_n557 = ~new_n553 & ~new_n556;
  assign new_n558 = ~new_n539 & new_n557;
  assign new_n559 = ~\ctable[0]  & ~new_n558;
  assign new_n560 = ~new_n438 & \totalcoeffs[0] ;
  assign new_n561 = ~\trailingones[1]  & \ctable[0] ;
  assign new_n562 = ~\totalcoeffs[0]  & new_n561;
  assign new_n563 = ~new_n560 & ~new_n562;
  assign new_n564 = ~new_n563 & \trailingones[0] ;
  assign new_n565 = new_n95 & new_n162;
  assign new_n566 = ~new_n564 & ~new_n565;
  assign new_n567 = ~\totalcoeffs[3]  & ~new_n566;
  assign new_n568 = \totalcoeffs[3]  & \ctable[0] ;
  assign new_n569 = ~new_n62 & ~new_n568;
  assign new_n570 = ~\totalcoeffs[0]  & ~new_n569;
  assign new_n571 = \totalcoeffs[3]  & new_n491;
  assign new_n572 = ~new_n570 & ~new_n571;
  assign new_n573 = ~\trailingones[0]  & ~new_n572;
  assign new_n574 = new_n554 & new_n561;
  assign new_n575 = ~new_n573 & ~new_n574;
  assign new_n576 = ~new_n567 & new_n575;
  assign new_n577 = ~new_n576 & \totalcoeffs[1] ;
  assign new_n578 = \totalcoeffs[2]  & \totalcoeffs[3] ;
  assign new_n579 = ~\totalcoeffs[2]  & ~new_n42;
  assign new_n580 = ~new_n435 & ~new_n579;
  assign new_n581 = ~new_n578 & ~new_n580;
  assign new_n582 = ~new_n581 & \ctable[0] ;
  assign new_n583 = new_n42 & new_n325;
  assign new_n584 = ~new_n582 & ~new_n583;
  assign new_n585 = ~\totalcoeffs[1]  & ~new_n584;
  assign new_n586 = \trailingones[0]  & new_n381;
  assign new_n587 = new_n578 & new_n586;
  assign new_n588 = ~new_n585 & ~new_n587;
  assign new_n589 = ~new_n577 & new_n588;
  assign new_n590 = ~\ctable[1]  & ~new_n589;
  assign new_n591 = ~new_n559 & ~new_n590;
  assign new_n592 = ~\ctable[2]  & ~new_n591;
  assign new_n593 = ~new_n305 & \totalcoeffs[2] ;
  assign new_n594 = ~\totalcoeffs[0]  & new_n593;
  assign new_n595 = ~\trailingones[1]  & new_n51;
  assign new_n596 = new_n60 & new_n595;
  assign new_n597 = ~new_n594 & ~new_n596;
  assign new_n598 = ~new_n597 & new_n216;
  assign new_n599 = new_n22 & new_n598;
  assign new_n600 = ~new_n592 & ~new_n599;
  assign new_n601 = ~\totalcoeffs[4]  & ~new_n600;
  assign new_n602 = new_n35 & new_n98;
  assign new_n603 = ~\ctable[2]  & \totalcoeffs[4] ;
  assign new_n604 = new_n216 & new_n603;
  assign new_n605 = new_n602 & new_n604;
  assign \ctoken_len[1]  = ~new_n601 & ~new_n605;
  assign new_n607 = new_n227 & new_n561;
  assign new_n608 = ~new_n283 & ~new_n607;
  assign new_n609 = ~\totalcoeffs[0]  & ~new_n608;
  assign new_n610 = ~\totalcoeffs[1]  & \ctable[0] ;
  assign new_n611 = ~new_n22 & ~new_n610;
  assign new_n612 = ~new_n110 & ~new_n255;
  assign new_n613 = ~new_n611 & new_n612;
  assign new_n614 = ~\totalcoeffs[2]  & new_n613;
  assign new_n615 = ~new_n440 & new_n332;
  assign new_n616 = ~new_n615 & new_n103;
  assign new_n617 = ~new_n614 & ~new_n616;
  assign new_n618 = ~new_n609 & new_n617;
  assign new_n619 = ~\trailingones[0]  & ~new_n618;
  assign new_n620 = ~new_n98 & ~new_n611;
  assign new_n621 = ~\totalcoeffs[2]  & new_n620;
  assign new_n622 = new_n139 & new_n203;
  assign new_n623 = ~new_n621 & ~new_n622;
  assign new_n624 = ~new_n623 & \trailingones[0] ;
  assign new_n625 = ~\ctable[0]  & ~new_n215;
  assign new_n626 = ~new_n568 & ~new_n625;
  assign new_n627 = new_n37 & new_n626;
  assign new_n628 = ~new_n624 & ~new_n627;
  assign new_n629 = ~\trailingones[1]  & ~new_n628;
  assign new_n630 = ~new_n261 & ~new_n568;
  assign new_n631 = ~new_n259 & ~new_n630;
  assign new_n632 = \totalcoeffs[1]  & new_n631;
  assign new_n633 = ~new_n173 & new_n310;
  assign new_n634 = new_n201 & new_n633;
  assign new_n635 = ~new_n632 & ~new_n634;
  assign new_n636 = ~new_n635 & \trailingones[1] ;
  assign new_n637 = ~new_n629 & ~new_n636;
  assign new_n638 = ~new_n619 & new_n637;
  assign new_n639 = ~\ctable[1]  & ~new_n638;
  assign new_n640 = ~\totalcoeffs[1]  & ~new_n125;
  assign new_n641 = ~new_n640 & \totalcoeffs[2] ;
  assign new_n642 = ~new_n329 & ~new_n641;
  assign new_n643 = ~new_n642 & \totalcoeffs[0] ;
  assign new_n644 = ~\ctable[1]  & ~new_n105;
  assign new_n645 = ~new_n644 & \totalcoeffs[2] ;
  assign new_n646 = ~new_n30 & \trailingones[0] ;
  assign new_n647 = ~new_n646 & new_n329;
  assign new_n648 = ~new_n645 & ~new_n647;
  assign new_n649 = ~new_n643 & new_n648;
  assign new_n650 = ~new_n649 & new_n386;
  assign new_n651 = ~new_n639 & ~new_n650;
  assign new_n652 = ~\ctable[2]  & ~new_n651;
  assign new_n653 = ~new_n146 & ~new_n506;
  assign new_n654 = ~\totalcoeffs[0]  & ~new_n653;
  assign new_n655 = ~new_n202 & ~new_n654;
  assign new_n656 = ~\trailingones[1]  & ~new_n655;
  assign new_n657 = new_n95 & new_n485;
  assign new_n658 = ~new_n656 & ~new_n657;
  assign new_n659 = ~\totalcoeffs[1]  & ~new_n658;
  assign new_n660 = new_n320 & new_n546;
  assign new_n661 = ~new_n659 & ~new_n660;
  assign new_n662 = ~new_n661 & new_n22;
  assign new_n663 = ~\ctable[1]  & new_n662;
  assign new_n664 = ~new_n652 & ~new_n663;
  assign new_n665 = ~\totalcoeffs[4]  & ~new_n664;
  assign new_n666 = new_n22 & new_n603;
  assign new_n667 = new_n425 & new_n666;
  assign \ctoken_len[2]  = ~new_n665 & ~new_n667;
  assign new_n669 = ~new_n140 & \totalcoeffs[4] ;
  assign new_n670 = new_n203 & new_n399;
  assign new_n671 = ~\ctable[2]  & ~new_n670;
  assign new_n672 = ~new_n640 & ~new_n671;
  assign new_n673 = ~\ctable[2]  & ~new_n140;
  assign new_n674 = ~new_n673 & \totalcoeffs[0] ;
  assign new_n675 = ~new_n103 & \ctable[2] ;
  assign new_n676 = ~\totalcoeffs[3]  & ~new_n57;
  assign new_n677 = ~new_n151 & \totalcoeffs[3] ;
  assign new_n678 = ~new_n677 & new_n215;
  assign new_n679 = ~new_n676 & ~new_n678;
  assign new_n680 = ~\totalcoeffs[2]  & ~new_n679;
  assign new_n681 = ~new_n356 & new_n399;
  assign new_n682 = ~\ctable[0]  & new_n681;
  assign new_n683 = ~\ctable[1]  & ~new_n586;
  assign new_n684 = ~\totalcoeffs[3]  & ~new_n683;
  assign new_n685 = ~new_n682 & ~new_n684;
  assign new_n686 = ~new_n680 & new_n685;
  assign new_n687 = ~new_n686 & \totalcoeffs[1] ;
  assign new_n688 = ~new_n410 & new_n215;
  assign new_n689 = ~new_n151 & ~new_n688;
  assign new_n690 = ~new_n689 & \totalcoeffs[3] ;
  assign new_n691 = ~new_n342 & \totalcoeffs[4] ;
  assign new_n692 = ~\totalcoeffs[3]  & ~new_n691;
  assign new_n693 = ~new_n690 & ~new_n692;
  assign new_n694 = ~\totalcoeffs[2]  & ~new_n693;
  assign new_n695 = ~\trailingones[0]  & ~new_n381;
  assign new_n696 = ~new_n695 & new_n676;
  assign new_n697 = \totalcoeffs[2]  & new_n696;
  assign new_n698 = ~new_n694 & ~new_n697;
  assign new_n699 = ~\totalcoeffs[1]  & ~new_n698;
  assign new_n700 = ~\ctable[0]  & ~new_n103;
  assign new_n701 = ~new_n700 & \ctable[1] ;
  assign new_n702 = ~new_n699 & ~new_n701;
  assign new_n703 = ~new_n687 & new_n702;
  assign new_n704 = ~new_n675 & new_n703;
  assign new_n705 = ~new_n674 & new_n704;
  assign new_n706 = ~new_n672 & new_n705;
  assign \ctoken_len[3]  = ~new_n669 & new_n706;
  assign new_n708 = ~\trailingones[0]  & ~new_n29;
  assign new_n709 = ~new_n25 & ~new_n708;
  assign new_n710 = ~new_n709 & new_n423;
  assign new_n711 = \totalcoeffs[2]  & new_n710;
  assign new_n712 = ~new_n430 & ~new_n711;
  assign new_n713 = ~\ctable[2]  & ~new_n712;
  assign \ctoken_len[4]  = new_n342 & new_n713;
endmodule


