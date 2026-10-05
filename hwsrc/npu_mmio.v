// KianV 的片上总线接 TinyQV 外设口的 MLP 引擎 tqvp_sohaib_npu。
// 引擎按「地址位加访问宽度」分辨操作（同一个地址 8 位读是取缩放后的结果、16 位读是取累加器），
// KianV 的总线读的时候不带宽度，所以这里摆一张按字编址的寄存器表，每个寄存器对应引擎的一种操作：
//
//   0x00 DAT    写  低 16 位是四个 int4 输入，高 16 位是四个 int4 权重；写一次乘加一次
//   0x04 ACC    写  累加器置成低 16 位（放偏置）；读  累加器，符号扩展
//   0x08 OUT    读  累加器按 QMUL、SHAMT 缩放并加零点后的 int4，符号扩展；要等引擎算六拍
//   0x0C SHAMT  写  右移位数，低 5 位
//   0x10 QMUL   写  量化乘数，低 15 位
//   0x14 CTRL   写  位 0 是 ReLU，位 4:1 是输出零点
//   0x1C ID     读  0x4E505531
`default_nettype none
module npu_mmio #(
    parameter [31:0] BASE = 32'h1070_0000
) (
    input  wire        clk,
    input  wire        resetn,
    input  wire        bus_valid_i,
    input  wire [31:0] bus_addr_i,
    input  wire [ 3:0] bus_wstrb_i,
    input  wire [31:0] bus_wdata_i,
    output reg  [31:0] bus_rdata_o,
    output reg         bus_ready_o,
    output wire        hit_o
);
  localparam [31:0] ID = 32'h4E50_5531;

  assign hit_o = bus_valid_i && (bus_addr_i[31:5] == BASE[31:5]);
  wire       wr  = |bus_wstrb_i;
  wire [2:0] sel = bus_addr_i[4:2];

  // 引擎那一侧：写的选通只给一拍（DAT 每一拍选通都乘加一次），读的选通顶到它答 data_ready
  reg  [ 5:0] p_addr;
  reg  [ 1:0] p_wn, p_rn;
  reg  [31:0] p_din;
  wire [31:0] p_dout;
  wire        p_ready;

  localparam [1:0] IDLE = 2'd0, WRITE = 2'd1, READ = 2'd2, DONE = 2'd3;
  reg [1:0] state;
  reg       narrow;

  always @(posedge clk) begin
    if (!resetn) begin
      state       <= IDLE;
      p_addr      <= 6'h00;
      p_wn        <= 2'b11;
      p_rn        <= 2'b11;
      p_din       <= 32'h0;
      narrow      <= 1'b0;
      bus_ready_o <= 1'b0;
      bus_rdata_o <= 32'h0;
    end else begin
      bus_ready_o <= 1'b0;
      case (state)
        IDLE:
        if (hit_o && !bus_ready_o) begin
          p_din       <= bus_wdata_i;
          bus_rdata_o <= 32'h0;
          state       <= DONE;
          if (wr) begin
            case (sel)
              3'd0: begin p_addr <= 6'h01; p_wn <= 2'b10; state <= WRITE; end
              3'd1: begin p_addr <= 6'h02; p_wn <= 2'b01; state <= WRITE; end
              3'd3: begin p_addr <= 6'h04; p_wn <= 2'b00; state <= WRITE; end
              3'd4: begin p_addr <= 6'h08; p_wn <= 2'b01; state <= WRITE; end
              3'd5: begin p_addr <= 6'h10; p_wn <= 2'b00; state <= WRITE; end
              default: ;
            endcase
          end else begin
            case (sel)
              3'd1: begin p_addr <= 6'h02; p_rn <= 2'b01; narrow <= 1'b0; state <= READ; end
              3'd2: begin p_addr <= 6'h02; p_rn <= 2'b00; narrow <= 1'b1; state <= READ; end
              3'd7: bus_rdata_o <= ID;
              default: ;
            endcase
          end
        end
        WRITE: begin
          p_wn  <= 2'b11;
          state <= DONE;
        end
        READ:
        if (p_ready) begin
          p_rn        <= 2'b11;
          bus_rdata_o <= narrow ? {{28{p_dout[3]}}, p_dout[3:0]} : {{16{p_dout[15]}}, p_dout[15:0]};
          state       <= DONE;
        end
        DONE: begin
          bus_ready_o <= 1'b1;
          state       <= IDLE;
        end
      endcase
    end
  end

  tqvp_sohaib_npu u_npu (
      .clk           (clk),
      .rst_n         (resetn),
      .ui_in         (8'h00),
      .uo_out        (),
      .address       (p_addr),
      .data_in       (p_din),
      .data_write_n  (p_wn),
      .data_read_n   (p_rn),
      .data_out      (p_dout),
      .data_ready    (p_ready),
      .user_interrupt()
  );
endmodule
`default_nettype wire
