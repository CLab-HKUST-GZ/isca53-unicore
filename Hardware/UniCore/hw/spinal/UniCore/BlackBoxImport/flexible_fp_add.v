
module flexible_fp_add (
    // *** Product *** //
    // * 4 Channel (W3/W4 Mode)
    input  wire [8 :0] E4M4_p0,
    input  wire [8 :0] E4M4_p1,
    input  wire [8 :0] E4M4_p2,
    input  wire [8 :0] E4M4_p3,
    // * 2 Channel (W8 Mode)
    input  wire [9 :0] E5M4_p0,
    input  wire [9 :0] E5M4_p1,
    // * 1 Channel (W16 Mode)
    input  wire [15:0] E8M7,

    // * Mode Select
    input  wire [1 :0] mode_sel,    // "00","01" for 4 channel, "10" for 2 channel, "11" for 1 channel

    // *** Partial Sum *** //
    // The PSum work as :
    //     {p0}, {p1}, {p2}, {p3} in 4 Channel Mode
    //     {p0,p1}, {p2,p3} in 2 Channel Mode
    //     {p0,p1,p2,p3} in 1 Channel Mode
    input  wire [17:0] PSum_p0,
    input  wire [17:0] PSum_p1,
    input  wire [17:0] PSum_p2,
    input  wire [17:0] PSum_p3,
    output wire [17:0] PSum_p0,
    output wire [17:0] PSum_p1,
    output wire [17:0] PSum_p2,
    output wire [17:0] PSum_p3
);



endmodule