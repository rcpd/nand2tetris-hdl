""""
Python implementation of the HACK architecture modelled after the Nand2Tetris HDL
NAND is a primitive implemented at the hardware level so need to define the logic ourselves
All subsequent gates can be expressed via increasingly complex abstractions of NAND
"""

# TODO: reduce multiple instantiations of classes where not required


class Gate(object):
    def __init__(self):
        # Gates values should only be initialized at runtime
        self.a = None
        self.b = None
        self.c = None
        self._in = None
        self.sel = None
        self.sel2 = None
        self.sel3 = None
        self._in8 = None
        self._in16 = None
        self.a16 = None
        self.b16 = None
        self.c16 = None
        self.d16 = None
        self.e16 = None
        self.f16 = None
        self.g16 = None
        self.h16 = None
        self.x = None
        self.y = None
        self.zx = None
        self.nx = None
        self.zy = None
        self.ny = None
        self.f = None
        self.no = None
        self.load = None
        self.inc = None
        self.reset = None
        self.addr3 = None
        self.addr6 = None

    def evaluate(self, a=None, b=None, c=None, _in=None, sel=None, sel2=None, sel3=None, _in8=None, _in16=None,
                 a16=None, b16=None, c16=None, d16=None, e16=None, f16=None, g16=None, h16=None, x=None, y=None,
                 zx=None, nx=None, zy=None, ny=None, f=None, no=None, load=None, inc=None, reset=None, addr3=None,
                 addr6=None):
        """
        validate input, None = uninitialized
        """
        # test flags
        one_bit_inputs = {"a": a, "b": b, "c": c, "_in": _in, "sel": sel, "zx": zx, "nx": nx, "zy": zy, "ny": ny,
                          "f": f, "no": no, "load": load, "inc": inc, "reset": reset}
        
        for flag in one_bit_inputs:
            if one_bit_inputs[flag] is not None:
                if type(one_bit_inputs[flag]) is not str:
                    one_bit_inputs[flag] = bin(one_bit_inputs[flag])
                if one_bit_inputs[flag] not in ("0b0", "0b1"):
                    raise RuntimeError("%s input must be 1 bit (value was %s)" % (flag, one_bit_inputs[flag]))
        
        sixteen_bit_inputs = {"_in16": _in16, "a16": a16, "b16": b16, "c16": c16, "d16": d16, "e16": e16, "f16": f16,
                              "g16": g16, "h16": h16, "x": x, "y": y}

        for flag in sixteen_bit_inputs:
            if sixteen_bit_inputs[flag] is not None:
                if type(sixteen_bit_inputs[flag]) is not str:
                    sixteen_bit_inputs[flag] = bin(sixteen_bit_inputs[flag])
                if int(sixteen_bit_inputs[flag], 2) < 0 or int(sixteen_bit_inputs[flag], 2) > 65535:
                    raise RuntimeError("%s input must be 16 bits (value was %s)" % (flag, sixteen_bit_inputs[flag]))
                
        # set inputs (either valid or None, not worth retesting for None)
        self.a = a
        self.b = b
        self.c = c
        self._in = _in
        self.sel = sel
        self.zx = zx
        self.nx = nx
        self.zy = zy
        self.ny = ny
        self.f = f
        self.no = no
        self._in16 = _in16
        self.a16 = a16
        self.b16 = b16
        self.c16 = c16
        self.d16 = d16
        self.e16 = e16
        self.f16 = f16
        self.g16 = g16
        self.h16 = h16
        self.x = x
        self.y = y
        self.load = load
        self.inc = inc
        self.reset = reset
        self.addr3 = addr3

        if sel2 is not None:
            if type(sel2) is not str:
                sel2 = bin(sel2)
            if sel2 not in ("0b00", "0b01", "0b10", "0b11"):
                raise RuntimeError("sel2 input must be 2 bits")
            self.sel2 = sel2
        
        if sel3 is not None:
            if type(sel3) is not str:
                sel3 = bin(sel3)
            if int(sel3, 2) < 0 or int(sel3, 2) > 7:
                raise RuntimeError("sel3 input must be 3 bits")
            self.sel3 = sel3
        
        if addr3 is not None:
            if type(addr3) is not str:
                addr3 = bin(addr3)
            if int(addr3, 2) < 0 or int(addr3, 2) > 7:
                raise RuntimeError("addr3 input must be 3 bits")
            self.addr3 = addr3

        if addr6 is not None:
            if type(addr6) is not str:
                addr6 = bin(addr6)
            if int(addr6, 2) < 0 or int(addr6, 2) > 77:
                raise RuntimeError("addr6 input must be 6 bits")
            self.addr6 = addr6

        if _in8 is not None:
            if type(_in8) is not str:
                _in8 = bin(_in8)
            if int(_in8, 2) < 0 or int(_in8, 2) > 255:
                raise RuntimeError("_in8 input must be 8 bits")
            self._in8 = _in8
            
        # run gate specific logic
        return self.calculate()

    def calculate(self):
        raise NotImplementedError


class NandGate(Gate):
    """
    For two 1 inputs return a 0 output, else return a 1 output
    """
    def calculate(self):
        # cast to int from binary string for comparison
        # endianness does not matter as only 1 bit returned
        if int(self.a, 2) and int(self.b, 2):
            return "0b0"
        else:
            return "0b1"


class NotGate(Gate):
    """
    For a single input, return the opposite

    CHIP Not {
        IN in;
        OUT out;

    PARTS:
        Nand(a=in, b=in, out=out);
    }
    """
    def calculate(self):
        """
        A single input/output can still have a one-to-many relationship with other gates
        In this case it is passed to both inputs of the NAND gate
        """
        # endianness does not matter as only 1 bit returned
        return NandGate().evaluate(a=self._in, b=self._in)


class Not16Gate(Gate):
    """
    NotGate but with two x 16 bit inputs and one 16 bit output, each bit is compared across both inputs
    CHIP Not16 {
        IN in[16];
        OUT out[16];

    PARTS:
        Not(in=in[0], out=out[0]);
        Not(in=in[1], out=out[1]);
        Not(in=in[2], out=out[2]);
        Not(in=in[3], out=out[3]);
        Not(in=in[4], out=out[4]);
        Not(in=in[5], out=out[5]);
        Not(in=in[6], out=out[6]);
        Not(in=in[7], out=out[7]);
        Not(in=in[8], out=out[8]);
        Not(in=in[9], out=out[9]);
        Not(in=in[10], out=out[10]);
        Not(in=in[11], out=out[11]);
        Not(in=in[12], out=out[12]);
        Not(in=in[13], out=out[13]);
        Not(in=in[14], out=out[14]);
        Not(in=in[15], out=out[15]);
    }
    """
    def calculate(self):
        byte_str = "0b"
        # will return an LSB endianness result which is then reversed to MSB for Python
        # HACK = LSB = 0000000000000001
        # PYBS = MSB = 1000000000000000
        for i in reversed(range(1, 17)):
            byte_str += NotGate().evaluate(_in="0b"+self._in16[i*-1])[2:]
        return byte_str


class AndGate(Gate):
    """
    For two 1 inputs return a 1 output, else return a 0 output

    CHIP And {
        IN a, b;
        OUT out;

    PARTS:
        Nand(a=a, b=b, out=aNand);
        Nand(a=aNand, b=aNand, out=out);
    }
    """
    def calculate(self):
        """
        Now we can combine a NAND with a NOT to get a regular AND gate
        """
        # endianness does not matter as only 1 bit returned
        nand_a = NandGate().evaluate(a=self.a, b=self.b)
        return NotGate().evaluate(_in=nand_a)


class And16Gate(Gate):
    """
    AndGate but with two x 16 bit inputs and one 16 bit output, each bit is compared across both inputs

    CHIP And {
        IN a, b;
        OUT out;

    PARTS:
        Nand(a=a, b=b, out=aNand);
        Nand(a=aNand, b=aNand, out=out);
    }
    """
    def calculate(self):
        byte_str = "0b"
        # will return an LSB endianness result which is then reversed to MSB for Python
        # HACK = LSB = 0000000000000001
        # PYBS = MSB = 1000000000000000
        for i in reversed(range(1, 17)):
            byte_str += AndGate().evaluate(a="0b"+self.a16[i*-1], b="0b"+self.b16[i*-1])[2:]
        return byte_str


class OrGate(Gate):
    """
    If either of the two inputs are 1 return a 1 output, else return a 0 output

    CHIP Or {
        IN a, b;
        OUT out;

    PARTS:
        Nand(a=a, b=a, out=aNand);
        Nand(a=b, b=b, out=bNand);
        Nand(a=aNand, b=bNand, out=out);
    }
    """
    def calculate(self):
        # endianness does not matter as only 1 bit returned
        nand_a = NandGate().evaluate(a=self.a, b=self.a)
        nand_b = NandGate().evaluate(a=self.b, b=self.b)
        return NandGate().evaluate(a=nand_a, b=nand_b)


class Or16Gate(Gate):
    """
    OrGate but with two x 16 bit inputs and one 16 bit output, each bit is compared across both inputs

    CHIP Or16 {
        IN a[16], b[16];
        OUT out[16];

    PARTS:
        Or(a=a[0], b=b[0], out=out[0]);
        Or(a=a[1], b=b[1], out=out[1]);
        Or(a=a[2], b=b[2], out=out[2]);
        Or(a=a[3], b=b[3], out=out[3]);
        Or(a=a[4], b=b[4], out=out[4]);
        Or(a=a[5], b=b[5], out=out[5]);
        Or(a=a[6], b=b[6], out=out[6]);
        Or(a=a[7], b=b[7], out=out[7]);
        Or(a=a[8], b=b[8], out=out[8]);
        Or(a=a[9], b=b[9], out=out[9]);
        Or(a=a[10], b=b[10], out=out[10]);
        Or(a=a[11], b=b[11], out=out[11]);
        Or(a=a[12], b=b[12], out=out[12]);
        Or(a=a[13], b=b[13], out=out[13]);
        Or(a=a[14], b=b[14], out=out[14]);
        Or(a=a[15], b=b[15], out=out[15]);
    }
    """
    def calculate(self):
        byte_str = "0b"
        # will return an LSB endianness result which is then reversed to MSB for Python
        # HACK = LSB = 0000000000000001
        # PYBS = MSB = 1000000000000000
        for i in reversed(range(1, 17)):
            byte_str += OrGate().evaluate(a="0b"+self.a16[i*-1], b="0b"+self.b16[i*-1])[2:]
        return byte_str


class Or8Way(Gate):
    """
    8 bit bus of 1 bit inputs, 1 bit output, if any bits 1 return 1, else 0

    CHIP Or8Way {
        IN in[8];
        OUT out;

    PARTS:
        Or(a=in[0], b=in[1], out=or1);
        Or(a=in[2], b=in[3], out=or2);
        Or(a=in[4], b=in[5], out=or3);
        Or(a=in[6], b=in[7], out=or4);
        Or(a=or1, b=or2, out=or5);
        Or(a=or3, b=or4, out=or6);
        Or(a=or5, b=or6, out=out);
    }
    """
    def calculate(self):
        # endianness does not matter as only 1 bit returned
        or0 = OrGate().evaluate(a="0b"+self._in8[-1], b="0b"+self._in8[-2])
        or1 = OrGate().evaluate(a="0b"+self._in8[-3], b="0b"+self._in8[-4])
        or2 = OrGate().evaluate(a="0b"+self._in8[-5], b="0b"+self._in8[-6])
        or3 = OrGate().evaluate(a="0b"+self._in8[-7], b="0b"+self._in8[-8])

        or4 = OrGate().evaluate(a=or0, b=or1)
        or5 = OrGate().evaluate(a=or2, b=or3)

        return OrGate().evaluate(a=or4, b=or5)


class XorGate(Gate):
    """
    If the two inputs are different return a 1 output, else return a 0 output

    CHIP Xor {
        IN a, b;
        OUT out;

    PARTS:
        Nand(a=a, b=b, out=aNand);
        Nand(a=a, b=aNand, out=bNand);
        Nand(a=aNand, b=b, out=cNand);
        Nand(a=bNand, b=cNand, out=out);
    }
    """
    def calculate(self):
        # endianness does not matter as only 1 bit returned
        nand_a = NandGate().evaluate(a=self.a, b=self.b)
        nand_b = NandGate().evaluate(a=self.a, b=nand_a)
        nand_c = NandGate().evaluate(a=nand_a, b=self.b)
        return NandGate().evaluate(a=nand_b, b=nand_c)


class NorGate(Gate):
    """
    If either of the two inputs are 1 return a 0 output, else return a 1 output
    CHIP Nor {
        IN a, b;
        OUT out;

    PARTS:
        Or(a=a,b=b,out=orOut);
        Not(in=orOut,out=out);
    }
    """
    def calculate(self):
        # endianness does not matter as only 1 bit returned
        _or = OrGate().evaluate(a=self.a, b=self.b)
        return NotGate().evaluate(_in=_or)


class XNorGate(Gate):
    """
    If the two inputs are different return a 0 output, else return a 1 output
    CHIP XNor {
        IN a, b;
        OUT out;

    PARTS:
        Xor(a=a,b=b,out=xorOut);
        Not(in=xorOut,out=out);
    }
    """
    def calculate(self):
        # endianness does not matter as only 1 bit returned
        _xor = XorGate().evaluate(a=self.a, b=self.b)
        return NotGate().evaluate(_in=_xor)


class Mux(Gate):
    """
    Select an output from two inputs, only chosen input will be emitted

    CHIP Mux {
        IN a, b, sel;
        OUT out;

    PARTS:
        Nand(a=sel, b=sel, out=aNand);
        Nand(a=b, b=sel, out=bNand);
        Nand(a=aNand, b=a, out=cNand);
        Nand(a=bNand, b=cNand, out=out);
    }
    """
    def calculate(self):
        # endianness does not matter as only 1 bit returned
        nand_a = NandGate().evaluate(a=self.sel, b=self.sel)
        nand_b = NandGate().evaluate(a=self.b, b=self.sel)
        nand_c = NandGate().evaluate(a=nand_a, b=self.a)
        return NandGate().evaluate(a=nand_b, b=nand_c)


class Mux16(Gate):
    """
    Mux but with two x 16 bit inputs and one 16 bit output, only chosen input will be emitted

    CHIP Mux16 {
        IN a[16], b[16], sel;
        OUT out[16];

    PARTS:
        Mux(a=a[0], b=b[0], sel=sel, out=out[0]);
        Mux(a=a[1], b=b[1], sel=sel, out=out[1]);
        Mux(a=a[2], b=b[2], sel=sel, out=out[2]);
        Mux(a=a[3], b=b[3], sel=sel, out=out[3]);
        Mux(a=a[4], b=b[4], sel=sel, out=out[4]);
        Mux(a=a[5], b=b[5], sel=sel, out=out[5]);
        Mux(a=a[6], b=b[6], sel=sel, out=out[6]);
        Mux(a=a[7], b=b[7], sel=sel, out=out[7]);
        Mux(a=a[8], b=b[8], sel=sel, out=out[8]);
        Mux(a=a[9], b=b[9], sel=sel, out=out[9]);
        Mux(a=a[10], b=b[10], sel=sel, out=out[10]);
        Mux(a=a[11], b=b[11], sel=sel, out=out[11]);
        Mux(a=a[12], b=b[12], sel=sel, out=out[12]);
        Mux(a=a[13], b=b[13], sel=sel, out=out[13]);
        Mux(a=a[14], b=b[14], sel=sel, out=out[14]);
        Mux(a=a[15], b=b[15], sel=sel, out=out[15]);
    }
    """
    def calculate(self):
        byte_str = "0b"
        # will return an LSB endianness result which is then reversed to MSB for Python
        # HACK = LSB = 0000000000000001
        # PYBS = MSB = 1000000000000000
        for i in reversed(range(1, 17)):
            byte_str += Mux().evaluate(a="0b"+self.a16[i*-1], b="0b"+self.b16[i*-1], sel=self.sel)[2:]
        return byte_str


class Mux4Way16(Gate):
    """
    Mux16 but with 4 x 16 bit inputs, one 16 bit output, two bit selector, only selected is emitted

    CHIP Mux4Way16 {
        IN a[16], b[16], c[16], d[16], sel[2];
        OUT out[16];

    PARTS:
        Mux16(a=a, b=b, sel=sel[0], out=muxAB);
        Mux16(a=c, b=d, sel=sel[0], out=muxCD);
        Mux16(a=muxAB, b=muxCD, sel=sel[1], out=out);
    }
    """
    def calculate(self):
        # endianness only matters for selector / result order
        mux16_ab = Mux16().evaluate(a16=self.a16, b16=self.b16, sel="0b"+self.sel2[-1])
        mux16_cd = Mux16().evaluate(a16=self.c16, b16=self.d16, sel="0b"+self.sel2[-1])
        return Mux16().evaluate(a16=mux16_ab, b16=mux16_cd, sel="0b"+self.sel2[-2])


class Mux8Way16(Gate):
    """
    Mux16 but with 8 x 16 bit inputs, one 16 bit output, 3 bit selector, only selected is emitted

    CHIP Mux8Way16 {
    IN a[16], b[16], c[16], d[16],
       e[16], f[16], g[16], h[16],
       sel[3];
    OUT out[16];

    PARTS:
        Mux4Way16(a=a, b=b, c=c, d=d, sel=sel[0..1], out=muxAD);
        Mux4Way16(a=e, b=f, c=g, d=h, sel=sel[0..1], out=muxEH);
        Mux16(a=muxAD, b=muxEH, sel=sel[2], out=out);
    }
    """
    def calculate(self):
        # endianness only matters for selector / result order
        mux4way16_ad = Mux4Way16().evaluate(a16=self.a16, b16=self.b16, c16=self.c16, d16=self.d16,
                                            sel2="0b"+self.sel3[-2]+self.sel3[-1])
        mux4way16_eh = Mux4Way16().evaluate(a16=self.e16, b16=self.f16, c16=self.g16, d16=self.h16,
                                            sel2="0b"+self.sel3[-2]+self.sel3[-1])
        return Mux16().evaluate(a16=mux4way16_ad, b16=mux4way16_eh, sel="0b"+self.sel3[-3])


class DMux(Mux):
    """
    Select one of two outputs, input passes through and unselected output is always 0

    CHIP DMux {
        IN in, sel;
        OUT a, b;

    PARTS:
        Nand(a=sel, b=sel, out=aNand);
        Nand(a=in, b=aNand, out=bNand);
        Nand(a=sel, b=in, out=cNand);
        Nand(a=bNand, b=bNand, out=a);
        Nand(a=cNand, b=cNand, out=b);
    }
    """
    def calculate(self):
        # endianness does not matter as only 2 x 1 bit returned?
        nand_a = NandGate().evaluate(a=self.sel, b=self.sel)
        nand_b = NandGate().evaluate(a=self._in, b=nand_a)
        nand_c = NandGate().evaluate(a=self.sel, b=self._in)
        return NandGate().evaluate(a=nand_b, b=nand_b), NandGate().evaluate(a=nand_c, b=nand_c)


class DMux4Way(DMux):
    """
    With a 2 bit selector choose one of four outputs, input passes through and unselected is always 0

    CHIP DMux4Way {
        IN in, sel[2];
        OUT a, b, c, d;

    PARTS:
        DMux(in=in, sel=sel[1], a=dIn0, b=dIn1);
        DMux(in=dIn0, sel=sel[0], a=a, b=b);
        DMux(in=dIn1, sel=sel[0], a=c, b=d);
    }
    """
    def calculate(self):
        # endianness only matters for selector / result order
        dmux_0_a, dmux_0_b = DMux().evaluate(_in=self._in, sel="0b"+self.sel2[-2])
        dmux_1_a, dmux_1_b = DMux().evaluate(_in=dmux_0_a, sel="0b"+self.sel2[-1])
        dmux_2_a, dmux_2_b = DMux().evaluate(_in=dmux_0_b, sel="0b"+self.sel2[-1])
        return dmux_1_a, dmux_1_b, dmux_2_a, dmux_2_b


class DMux8Way(DMux):
    """
    With a 3 bit selector choose one of 8 outputs, input passes through and unselected is always 0

    CHIP DMux8Way {
        IN in, sel[3];
        OUT a, b, c, d, e, f, g, h;

    PARTS:
        DMux(in=in, sel=sel[2], a=dIn0, b=dIn1);
        DMux4Way(in=dIn0, sel=sel[0..1], a=a, b=b, c=c, d=d);
        DMux4Way(in=dIn1, sel=sel[0..1], a=e, b=f, c=g, d=h);
    }
    """
    def calculate(self):
        # endianness only matters for selector / result order
        dmux_a, dmux_b = DMux().evaluate(_in=self._in, sel="0b"+self.sel3[-3])
        dmux4_0 = DMux4Way().evaluate(_in=dmux_a, sel2="0b"+self.sel3[-2]+self.sel3[-1])
        dmux4_1 = DMux4Way().evaluate(_in=dmux_b, sel2="0b"+self.sel3[-2]+self.sel3[-1])
        return dmux4_0 + dmux4_1


class HalfAdder(Gate):
    """
    Computes the sum of 2 x 1 bit inputs, output carry bit & sum bit

    CHIP HalfAdder {
        IN a, b;    // 1-bit inputs
        OUT sum,    // Right bit of a + b
            carry;  // Left bit of a + b

    PARTS:
        Xor(a=a, b=b, out=sum);
        And(a=a, b=b, out=carry);
    }
    """
    def calculate(self):
        carry = AndGate().evaluate(a=self.a, b=self.b)
        _sum = XorGate().evaluate(a=self.a, b=self.b)
        return carry, _sum


class FullAdder(Gate):
    """
    Computes the sum of 3 x 1 bit inputs, output carry bit & sum bit

    CHIP FullAdder {
        IN a, b, c;  // 1-bit inputs
        OUT sum,     // Right bit of a + b + c
            carry;   // Left bit of a + b + c

    PARTS:
        HalfAdder(a=a, b=b, sum=sumAB, carry=carryAB);
        HalfAdder(a=c, b=sumAB, sum=sum, carry=carryABC);
        Or(a=carryABC, b=carryAB, out=carry);
    }
    """
    def calculate(self):
        carry_ab, sum_ab = HalfAdder().evaluate(a=self.a, b=self.b)
        carry_abc, _sum = HalfAdder().evaluate(a=self.c, b=sum_ab)
        carry = OrGate().evaluate(a=carry_abc, b=carry_ab)
        return carry, _sum


class Add16(Gate):
    """
    Adds two 16-bit values and output 16 bit result, the most significant carry bit is ignored

    CHIP Add16 {
        IN a[16], b[16];
        OUT out[16];

    PARTS:
        HalfAdder(a=a[0], b=b[0], sum=out[0], carry=carry0);
        FullAdder(a=a[1], b=b[1], c=carry0, sum=out[1], carry=carry1);
        FullAdder(a=a[2], b=b[2], c=carry1, sum=out[2], carry=carry2);
        FullAdder(a=a[3], b=b[3], c=carry2, sum=out[3], carry=carry3);
        FullAdder(a=a[4], b=b[4], c=carry3, sum=out[4], carry=carry4);
        FullAdder(a=a[5], b=b[5], c=carry4, sum=out[5], carry=carry5);
        FullAdder(a=a[6], b=b[6], c=carry5, sum=out[6], carry=carry6);
        FullAdder(a=a[7], b=b[7], c=carry6, sum=out[7], carry=carry7);
        FullAdder(a=a[8], b=b[8], c=carry7, sum=out[8], carry=carry8);
        FullAdder(a=a[9], b=b[9], c=carry8, sum=out[9], carry=carry9);
        FullAdder(a=a[10], b=b[10], c=carry9, sum=out[10], carry=carry10);
        FullAdder(a=a[11], b=b[11], c=carry10, sum=out[11], carry=carry11);
        FullAdder(a=a[12], b=b[12], c=carry11, sum=out[12], carry=carry12);
        FullAdder(a=a[13], b=b[13], c=carry12, sum=out[13], carry=carry13);
        FullAdder(a=a[14], b=b[14], c=carry13, sum=out[14], carry=carry14);
        FullAdder(a=a[15], b=b[15], c=carry14, sum=out[15], carry=carry15); // carry15 goes nowhere
    }
    """
    def calculate(self):
        _sum = ["X"] * 16
        carry = ["X"] * 16
        carry[-1], _sum[-1] = HalfAdder().evaluate(a="0b"+self.a16[-1], b="0b"+self.b16[-1])
        for i in range(2, 17):  # -2 to -16
            i = i * -1
            carry[i], _sum[i] = FullAdder().evaluate(a="0b"+self.a16[i], b="0b"+self.b16[i], c=carry[i+1])

        _sum_result = "0b"
        for bit in _sum:
            _sum_result += bit.replace("0b", "")

        return _sum_result


class Inc16(Gate):
    """
    Increment a 16 bit number

    CHIP Inc16 {
        IN in[16];
        OUT out[16];

    PARTS:
        Add16(a=in, b[0]=true, b[1..15]=false, out=out);
    }
    """
    def calculate(self):
        return Add16().evaluate(a16=self._in16, b16="0b0000000000000001")


class ALU(Gate):
    """
    ALU (Arithmetic Logic Unit) Computes one of the following functions:
        x+y, x-y, y-x, 0, 1, -1, x, y, -x, -y, !x, !y,
        x+1, y+1, x-1, y-1, x&y, x|y on two 16-bit inputs,

    // if (zx == 1) set x = 0        // 16-bit constant
    // if (nx == 1) set x = !x       // bitwise not
    // if (zy == 1) set y = 0        // 16-bit constant
    // if (ny == 1) set y = !y       // bitwise not
    // if (f == 1)  set out = x + y  // integer 2's complement addition
    // if (f == 0)  set out = x & y  // bitwise and
    // if (no == 1) set out = !out   // bitwise not
    // if (out == 0) set zr = 1
    // if (out < 0) set ng = 1

    CHIP ALU {
    IN
        x[16], y[16],  // 16-bit inputs
        zx, nx, zy, ny, f, no; // 1 bit flags

    OUT
        out[16], // 16-bit output
        zr, ng; // 1 bit flags

    PARTS:
        //zx/zy (1=zero input)
        Mux16(a=x, b=false, sel=zx, out=xZ);
        Mux16(a=y, b=false, sel=zy, out=yZ);
    
        //nx/ny (1=not input)
        Not16(in=xZ, out=xZnot);
        Not16(in=yZ, out=yZnot);
        Mux16(a=xZ, b=xZnot, sel=nx, out=xZN);
        Mux16(a=yZ, b=yZnot, sel=ny, out=yZN);
    
        //f (0=and, 1=add)
        Add16(a=xZN, b=yZN, out=xyZNadd);
        And16(a=xZN, b=yZN, out=xyZNand);
        Mux16(a=xyZNand, b=xyZNadd, sel=f, out=xyZNF);
    
        //no (1=not output)
        Not16(in=xyZNF, out=xyZNFnot);
        Mux16(a=xyZNF, b=xyZNFnot, sel=no, out[0..7]=xyZNFN1,
            out[8..15]=xyZNFN2, out=out, out[15]=ng); // ng=MSB (nostat compatible)
    
        //zr (0=[result==0], 1=[result!=0]
        Or8Way(in=xyZNFN1, out=Or81);
        Or8Way(in=xyZNFN2, out=Or82);
        Or(a=Or81, b=Or82, out=zrOr);
        Not(in=zrOr, out=zr);
    }
    """
    def __init__(self):
        super().__init__()
        # output flags only, not checked in evaluate()
        self.zr = None
        self.ng = None

    def calculate(self):
        # zx/zy (1=zero input)
        x_z = Mux16().evaluate(a16=self.x, b16="0b0000000000000000", sel=self.zx)
        y_z = Mux16().evaluate(a16=self.y, b16="0b0000000000000000", sel=self.zy)

        # nx/ny (1=not input)
        x_n = Not16Gate().evaluate(_in16=x_z)
        y_n = Not16Gate().evaluate(_in16=y_z)
        x_zn = Mux16().evaluate(a16=x_z, b16=x_n, sel=self.nx)
        y_zn = Mux16().evaluate(a16=y_z, b16=y_n, sel=self.ny)

        # (0=and, 1=add)
        xy_zn_add = Add16().evaluate(a16=x_zn, b16=y_zn)
        xy_zn_and = And16Gate().evaluate(a16=x_zn, b16=y_zn)
        xy_znf = Mux16().evaluate(a16=xy_zn_and, b16=xy_zn_add, sel=self.f)

        # no (1=not) // ng = MSB
        xy_znf_not = Not16Gate().evaluate(_in16=xy_znf)
        result = Mux16().evaluate(a16=xy_znf, b16=xy_znf_not, sel=self.no)
        self.ng = "0b" + result[-16]

        # zr (1 = result==0) // endianness doesn't matter in this instance
        result_1 = result[-16:-8]  # out[0..7]=xyZNFN1
        result_2 = result[-8:]  # out[8..15]=xyZNFN2
        or8_1 = Or8Way().evaluate(_in8=result_1)
        or8_2 = Or8Way().evaluate(_in8=result_2)
        zr_or = OrGate().evaluate(a=or8_1, b=or8_2)
        self.zr = NotGate().evaluate(_in=zr_or)

        return result, self.zr, self.ng


class DFF(Gate):
    """
    1 bit register, store new value if load else emit previous value

    // No HDL, implemented in Java on the course
    // DFF(in=Mout,out=Dout,out=out);
    """
    def __init__(self):
        super().__init__()
        self.r_nor = "0b0"  # Q = in if load
        self.s_nor = "0b1"  # !Q

    def calculate(self):
        # reset=(in=0 & load=1)
        load1 = AndGate().evaluate(a=self.load, b="0b1")
        in0 = NorGate().evaluate(a=self._in, b="0b0")
        reset = AndGate().evaluate(a=load1, b=in0)

        s_and = AndGate().evaluate(a=self._in, b=self.load)
        r_and = AndGate().evaluate(a=self.load, b=reset)
        self.s_nor = NorGate().evaluate(a=s_and, b=self.r_nor)
        self.r_nor = NorGate().evaluate(a=self.s_nor, b=r_and)

        # print("S: %s" % s_and, "R: %s" % r_and, "--", "Q: %s" % self.r_nor, "!Q: %s" % self.s_nor)
        if self.r_nor == "0b1" and self.s_nor == "0b1":
            raise RuntimeError("DFF failed, r_nor/s_nor cannot both be 0b1: %s %s" % (self.r_nor, self.s_nor))
        return self.r_nor, self.s_nor


class Bit(Gate):
    """
    1 bit register, if load emit in else previous value

    CHIP Bit {
        IN in, load;
        OUT out;

    PARTS:
        Mux(a=Dout,b=in,sel=load,out=Mout);
        DFF(in=Mout,out=Dout,out=out);
    }
    """
    def __init__(self):
        super().__init__()
        self.dff = DFF()
        self.d_out = "0b0"

    def calculate(self):
        m_out = Mux().evaluate(a=self.d_out, b=self._in, sel=self.load)
        self.d_out = self.dff.evaluate(_in=m_out, load=self.load)[0]
        return self.d_out


class Register(Gate):
    """
    16 bit register, if load emit in else previous value

    CHIP Register {
        IN in[16], load;
        OUT out[16];

    PARTS:
        Bit(in=in[0], load=load, out=out[0]);
        Bit(in=in[1], load=load, out=out[1]);
        Bit(in=in[2], load=load, out=out[2]);
        Bit(in=in[3], load=load, out=out[3]);
        Bit(in=in[4], load=load, out=out[4]);
        Bit(in=in[5], load=load, out=out[5]);
        Bit(in=in[6], load=load, out=out[6]);
        Bit(in=in[7], load=load, out=out[7]);
        Bit(in=in[8], load=load, out=out[8]);
        Bit(in=in[9], load=load, out=out[9]);
        Bit(in=in[10], load=load, out=out[10]);
        Bit(in=in[11], load=load, out=out[11]);
        Bit(in=in[12], load=load, out=out[12]);
        Bit(in=in[13], load=load, out=out[13]);
        Bit(in=in[14], load=load, out=out[14]);
        Bit(in=in[15], load=load, out=out[15]);
    }
    """
    def __init__(self):
        super().__init__()
        self.bit0 = Bit()
        self.bit1 = Bit()
        self.bit2 = Bit()
        self.bit3 = Bit()
        self.bit4 = Bit()
        self.bit5 = Bit()
        self.bit6 = Bit()
        self.bit7 = Bit()
        self.bit8 = Bit()
        self.bit9 = Bit()
        self.bit10 = Bit()
        self.bit11 = Bit()
        self.bit12 = Bit()
        self.bit13 = Bit()
        self.bit14 = Bit()
        self.bit15 = Bit()

    def calculate(self):
        # can't use range as Register has to save state
        bit0 = self.bit0.evaluate(_in="0b"+self._in16[2], load=self.load)
        bit1 = self.bit1.evaluate(_in="0b"+self._in16[3], load=self.load)
        bit2 = self.bit2.evaluate(_in="0b"+self._in16[4], load=self.load)
        bit3 = self.bit3.evaluate(_in="0b"+self._in16[5], load=self.load)
        bit4 = self.bit4.evaluate(_in="0b"+self._in16[6], load=self.load)
        bit5 = self.bit5.evaluate(_in="0b"+self._in16[7], load=self.load)
        bit6 = self.bit6.evaluate(_in="0b"+self._in16[8], load=self.load)
        bit7 = self.bit7.evaluate(_in="0b"+self._in16[9], load=self.load)
        bit8 = self.bit8.evaluate(_in="0b"+self._in16[10], load=self.load)
        bit9 = self.bit9.evaluate(_in="0b"+self._in16[11], load=self.load)
        bit10 = self.bit10.evaluate(_in="0b"+self._in16[12], load=self.load)
        bit11 = self.bit11.evaluate(_in="0b"+self._in16[13], load=self.load)
        bit12 = self.bit12.evaluate(_in="0b"+self._in16[14], load=self.load)
        bit13 = self.bit13.evaluate(_in="0b"+self._in16[15], load=self.load)
        bit14 = self.bit14.evaluate(_in="0b"+self._in16[16], load=self.load)
        bit15 = self.bit15.evaluate(_in="0b"+self._in16[17], load=self.load)

        return "0b" + bit0[2:] + bit1[2:] + bit2[2:] + bit3[2:] + bit4[2:] + bit5[2:] \
               + bit6[2:] + bit7[2:] + bit8[2:] + bit9[2:] + bit10[2:] + bit11[2:] \
               + bit12[2:] + bit13[2:] + bit14[2:] + bit15[2:]


class PC(Gate):
    """
    A 16-bit counter with load and reset control bits.

    // if      (reset[t] == 1) out[t+1] = 0
    // else if (load[t] == 1)  out[t+1] = in[t]
    // else if (inc[t] == 1)   out[t+1] = out[t] + 1  (integer addition)
    // else                    out[t+1] = out[t]

    CHIP PC {
        IN in[16],load,inc,reset;
        OUT out[16];

    PARTS:
        Inc16(in=feedback, out=pc);
        Mux16(a=feedback, b=pc, sel=inc, out=w0);
        Mux16(a=w0, b=in, sel=load, out=w1);
        Mux16(a=w1, b=false, sel=reset, out=cout);
        Register(in=cout, load=true, out=out, out=feedback);
    }
    """
    def __init__(self):
        super().__init__()
        self.feedback = "0b0000000000000000"

    def calculate(self):
        pc_inc = Inc16().evaluate(_in16=self.feedback)
        mux16_w0 = Mux16().evaluate(a16=self.feedback, b16=pc_inc, sel=self.inc)
        mux16_w1 = Mux16().evaluate(a16=mux16_w0, b16=self._in16, sel=self.load)
        mux16_cout = Mux16().evaluate(a16=mux16_w1, b16="0b0000000000000000", sel=self.reset)
        self.feedback = Register().evaluate(_in16=mux16_cout, load="0b1")
        return self.feedback


class RAM8(Gate):
    """
    Memory of 8 registers, each 16 bit-wide. 
    Out holds the value stored at the memory location specified by address.
    If load==1, then the in value is loaded into the memory location specified by address
    
    CHIP RAM8 {
        IN in[16], load, address[3];
        OUT out[16];
    
    PARTS:
        DMux8Way(in=load, sel=address, a=r0, b=r1, c=r2, d=r3, e=r4, f=r5, g=r6, h=r7);
        Register(in=in, load=r0, out=r0Out);
        Register(in=in, load=r1, out=r1Out);
        Register(in=in, load=r2, out=r2Out);
        Register(in=in, load=r3, out=r3Out);
        Register(in=in, load=r4, out=r4Out);
        Register(in=in, load=r5, out=r5Out);
        Register(in=in, load=r6, out=r6Out);
        Register(in=in, load=r7, out=r7Out);
        Mux8Way16(a=r0Out, b=r1Out, c=r2Out, d=r3Out, e=r4Out, f=r5Out, g=r6Out, h=r7Out, sel=address, out=out);
    }
    """
    def __init__(self):
        super().__init__()
        self.r0 = Register()
        self.r1 = Register()
        self.r2 = Register()
        self.r3 = Register()
        self.r4 = Register()
        self.r5 = Register()
        self.r6 = Register()
        self.r7 = Register()

    def calculate(self):
        dmux8w = DMux8Way().evaluate(_in=self.load, sel3=self.addr3)
        r0 = self.r0.evaluate(_in16=self._in16, load=dmux8w[0])
        r1 = self.r1.evaluate(_in16=self._in16, load=dmux8w[1])
        r2 = self.r2.evaluate(_in16=self._in16, load=dmux8w[2])
        r3 = self.r3.evaluate(_in16=self._in16, load=dmux8w[3])
        r4 = self.r4.evaluate(_in16=self._in16, load=dmux8w[4])
        r5 = self.r5.evaluate(_in16=self._in16, load=dmux8w[5])
        r6 = self.r6.evaluate(_in16=self._in16, load=dmux8w[6])
        r7 = self.r7.evaluate(_in16=self._in16, load=dmux8w[7])
        # print("class", r0, r1, r2, r3, r4, r5, r6, r7)
        return Mux8Way16().evaluate(a16=r0, b16=r1, c16=r2, d16=r3, e16=r4, f16=r5, g16=r6, h16=r7, sel3=self.addr3)


class RAM64(Gate):
    """
    Memory of 64 registers, each 16 bit-wide.
    Out holds the value stored at the memory location specified by address.
    If load==1, then the in value is loaded into the memory location specified by address

    CHIP RAM64 {
        IN in[16], load, address[6];
        OUT out[16];

    PARTS:
        DMux8Way(in=load, sel=address[0..2], a=r0, b=r1, c=r2, d=r3, e=r4, f=r5, g=r6, h=r7);
        RAM8(in=in, load=r0, address=address[3..5], out=r0Out);
        RAM8(in=in, load=r1, address=address[3..5], out=r1Out);
        RAM8(in=in, load=r2, address=address[3..5], out=r2Out);
        RAM8(in=in, load=r3, address=address[3..5], out=r3Out);
        RAM8(in=in, load=r4, address=address[3..5], out=r4Out);
        RAM8(in=in, load=r5, address=address[3..5], out=r5Out);
        RAM8(in=in, load=r6, address=address[3..5], out=r6Out);
        RAM8(in=in, load=r7, address=address[3..5], out=r7Out);
        Mux8Way16(a=r0Out, b=r1Out, c=r2Out, d=r3Out, e=r4Out, f=r5Out, g=r6Out, h=r7Out, sel=address[0..2], out=out);
    }
    """
    def __init__(self):
        super().__init__()
        self.r0 = RAM8()
        self.r1 = RAM8()
        self.r2 = RAM8()
        self.r3 = RAM8()
        self.r4 = RAM8()
        self.r5 = RAM8()
        self.r6 = RAM8()
        self.r7 = RAM8()

    def calculate(self):
        dmux8w = DMux8Way().evaluate(_in=self.load, sel3="0b"+self.addr6[-3:])
        r0 = self.r0.evaluate(_in16=self._in16, load=dmux8w[0], addr3=self.addr6[:-3])
        r1 = self.r1.evaluate(_in16=self._in16, load=dmux8w[1], addr3=self.addr6[:-3])
        r2 = self.r2.evaluate(_in16=self._in16, load=dmux8w[2], addr3=self.addr6[:-3])
        r3 = self.r3.evaluate(_in16=self._in16, load=dmux8w[3], addr3=self.addr6[:-3])
        r4 = self.r4.evaluate(_in16=self._in16, load=dmux8w[4], addr3=self.addr6[:-3])
        r5 = self.r5.evaluate(_in16=self._in16, load=dmux8w[5], addr3=self.addr6[:-3])
        r6 = self.r6.evaluate(_in16=self._in16, load=dmux8w[6], addr3=self.addr6[:-3])
        r7 = self.r7.evaluate(_in16=self._in16, load=dmux8w[7], addr3=self.addr6[:-3])
        print("class", r0, r1, r2, r3, r4, r5, r6, r7)
        return Mux8Way16().evaluate(a16=r0, b16=r1, c16=r2, d16=r3, e16=r4, f16=r5, g16=r6, h16=r7,
                                    sel3="0b"+self.addr6[-3:])


def input_unit_test():
    """
    Test input sizes: catch RuntimeException(s)
    """
    # input size unit test
    _gate = Gate()
    test = []
    for i in range(0, 9):
        try:
            if i == 0:
                # non-binary value
                test.append(_gate.evaluate(a="foobar"))  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 1:
                # 1 bit upper bound
                test.append(_gate.evaluate(a="0b10"))  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 2:
                # 2 bit lower bound
                test.append(_gate.evaluate(sel2="0b1"))  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 3:
                # 2 bit upper bound
                test.append(_gate.evaluate(sel2="0b100"))  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 4:
                # 3 bit lower bound
                test.append(_gate.evaluate(sel3="0b11"))  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 5:
                # 3 bit upper bound
                test.append(_gate.evaluate(sel3="0b1000"))  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 6:
                # 8 bit lower bound
                test.append(_gate.evaluate(_in8="0b1111111"))  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 7:
                # 8 bit upper bound
                test.append(_gate.evaluate(_in8="0b100000000"))  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 8:
                # 16 bit lower bound
                test.append(_gate.evaluate(a16="0b111111111111111"))  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
            if i == 9:
                # 16 bit upper bound
                test.append(_gate.evaluate(a16="0b10000000000000000"))  # fail
                raise Exception("Unit test failed (continued where it should have excepted)")
        except RuntimeError or NotImplementedError:
            continue
    assert test == []


def main():
    """
    Sanity check our truth tables for each gate as implemented
    """
    _nand = NandGate()
    _not = NotGate()
    _not16 = Not16Gate()
    _and = AndGate()
    _and16 = And16Gate()
    _or = OrGate()
    _or16 = Or16Gate()
    _or8way = Or8Way()
    _xor = XorGate()
    _nor = NorGate()
    _xnor = XNorGate()
    _mux = Mux()
    _mux16 = Mux16()
    _mux4way16 = Mux4Way16()
    _mux8way16 = Mux8Way16()
    _dmux = DMux()
    _dmux4way = DMux4Way()
    _dmux8way = DMux8Way()
    _halfAdder = HalfAdder()
    _fullAdder = FullAdder()
    _add16 = Add16()
    _inc16 = Inc16()
    _alu = ALU()
    _dff = DFF()
    _bit = Bit()
    _register = Register()
    _pc = PC()
    _ram8 = RAM8()
    _ram64 = RAM64()

    input_unit_test()

    # For two 1 inputs return a 1 output, else return a 1 output
    assert _nand.evaluate(a="0b1", b="0b1") == "0b0"
    assert _nand.evaluate(a="0b1", b="0b0") == "0b1"
    assert _nand.evaluate(a="0b0", b="0b1") == "0b1"
    assert _nand.evaluate(a="0b0", b="0b0") == "0b1"

    # For a single input, return the opposite
    assert _not.evaluate(_in="0b1") == "0b0"
    assert _not.evaluate(_in="0b0") == "0b1"

    # NotGate but with two x 16 bit inputs and one 16 bit output, each bit is compared across both inputs
    assert _not16.evaluate(_in16="0b0000000000000000") == "0b1111111111111111"
    assert _not16.evaluate(_in16="0b1111111111111111") == "0b0000000000000000"
    assert _not16.evaluate(_in16="0b0000001111000000") == "0b1111110000111111"

    # For two 1 inputs return a 1 output, else return a 0 output
    assert _and.evaluate(a="0b1", b="0b1") == "0b1"
    assert _and.evaluate(a="0b1", b="0b0") == "0b0"
    assert _and.evaluate(a="0b0", b="0b1") == "0b0"
    assert _and.evaluate(a="0b0", b="0b0") == "0b0"

    # AndGate but with two x 16 bit inputs and one 16 bit output, each bit is compared across both inputs
    assert _and16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000") == "0b0000000000000000"
    assert _and16.evaluate(a16="0b1111111111111111", b16="0b1111111111111111") == "0b1111111111111111"
    assert _and16.evaluate(a16="0b0000001111000000", b16="0b0000000000000000") == "0b0000000000000000"
    assert _and16.evaluate(a16="0b0000001111000000", b16="0b0000001111000000") == "0b0000001111000000"

    # If either of the two inputs are 1 return a 1 output, else return a 0 output
    assert _or.evaluate(a="0b1", b="0b1") == "0b1"
    assert _or.evaluate(a="0b1", b="0b0") == "0b1"
    assert _or.evaluate(a="0b0", b="0b1") == "0b1"
    assert _or.evaluate(a="0b0", b="0b0") == "0b0"

    # If either of the two inputs are 1 return a 0 output, else return a 1 output
    assert _nor.evaluate(a="0b1", b="0b1") == "0b0"
    assert _nor.evaluate(a="0b1", b="0b0") == "0b0"
    assert _nor.evaluate(a="0b0", b="0b1") == "0b0"
    assert _nor.evaluate(a="0b0", b="0b0") == "0b1"

    # OrGate but with two x 16 bit inputs and one 16 bit output, each bit is compared across both inputs
    assert _or16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000") == "0b0000000000000000"
    assert _or16.evaluate(a16="0b1111111111111111", b16="0b1111111111111111") == "0b1111111111111111"
    assert _or16.evaluate(a16="0b0000001111000000", b16="0b0000000000000000") == "0b0000001111000000"
    assert _or16.evaluate(a16="0b1111000000000000", b16="0b0000000000000000") == "0b1111000000000000"

    # 8 bit bus of 1 bit inputs, 1 bit output, if any bits 1 return 1, else 0
    assert _or8way.evaluate(_in8="0b11111111") == "0b1"
    assert _or8way.evaluate(_in8="0b00011000") == "0b1"
    assert _or8way.evaluate(_in8="0b00000000") == "0b0"

    # If the two inputs are different return a 1 output, else return a 0 output
    assert _xor.evaluate(a="0b1", b="0b1") == "0b0"
    assert _xor.evaluate(a="0b1", b="0b0") == "0b1"
    assert _xor.evaluate(a="0b0", b="0b1") == "0b1"
    assert _xor.evaluate(a="0b0", b="0b0") == "0b0"

    # If the two inputs are different return a 0 output, else return a 1 output
    assert _xnor.evaluate(a="0b1", b="0b1") == "0b1"
    assert _xnor.evaluate(a="0b1", b="0b0") == "0b0"
    assert _xnor.evaluate(a="0b0", b="0b1") == "0b0"
    assert _xnor.evaluate(a="0b0", b="0b0") == "0b1"

    # Select an output from two inputs, only chosen input will be emitted
    assert _mux.evaluate(a="0b1", b="0b0", sel="0b0") == "0b1"
    assert _mux.evaluate(a="0b1", b="0b0", sel="0b1") == "0b0"
    assert _mux.evaluate(a="0b0", b="0b1", sel="0b0") == "0b0"
    assert _mux.evaluate(a="0b0", b="0b1", sel="0b1") == "0b1"

    # Mux but with two x 16 bit inputs and one 16 bit output, each bit is compared across both inputs
    assert _mux16.evaluate(a16="0b1111111111111111", b16="0b0000000000000000", sel="0b0") == "0b1111111111111111"
    assert _mux16.evaluate(a16="0b1111111111111111", b16="0b0000000000000000", sel="0b1") == "0b0000000000000000"
    assert _mux16.evaluate(a16="0b0000000000000000", b16="0b1111111111111111", sel="0b0") == "0b0000000000000000"
    assert _mux16.evaluate(a16="0b0000000000000000", b16="0b1111111111111111", sel="0b1") == "0b1111111111111111"

    # Mux16 but with 4 x 16 bit inputs, one 16 bit output, two bit selector, only selected is emitted
    assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", sel2="0b00") == "0b0000000000000000"
    assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", sel2="0b01") == "0b0000000000000000"
    assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", sel2="0b10") == "0b0000000000000000"
    assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", sel2="0b11") == "0b0000000000000000"
    assert _mux4way16.evaluate(a16="0b1111111111111111", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", sel2="0b00") == "0b1111111111111111"
    assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b1111111111111111", c16="0b0000000000000000", d16="0b0000000000000000", sel2="0b01") == "0b1111111111111111"
    assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b1111111111111111", d16="0b0000000000000000", sel2="0b10") == "0b1111111111111111"
    assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b1111111111111111", sel2="0b11") == "0b1111111111111111"

    # Mux16 but with 8 x 16 bit inputs, one 16 bit output, 3 bit selector, only selected is emitted
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b000") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b001") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b010") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b011") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b100") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b101") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b110") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b111") == "0b0000000000000000"
    assert _mux8way16.evaluate(a16="0b1111111111111111", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b000") == "0b1111111111111111"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b1111111111111111", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b001") == "0b1111111111111111"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b1111111111111111", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b010") == "0b1111111111111111"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b1111111111111111", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b011") == "0b1111111111111111"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b1111111111111111", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b100") == "0b1111111111111111"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b1111111111111111", g16="0b0000000000000000", h16="0b0000000000000000", sel3="0b101") == "0b1111111111111111"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b1111111111111111", h16="0b0000000000000000", sel3="0b110") == "0b1111111111111111"
    assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000", d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000", g16="0b0000000000000000", h16="0b1111111111111111", sel3="0b111") == "0b1111111111111111"

    # Select one of two outputs, input passes through and unselected output is always 0
    assert _dmux.evaluate(_in="0b0", sel="0b0") == ("0b0", "0b0")
    assert _dmux.evaluate(_in="0b0", sel="0b1") == ("0b0", "0b0")
    assert _dmux.evaluate(_in="0b1", sel="0b0") == ("0b1", "0b0")
    assert _dmux.evaluate(_in="0b1", sel="0b1") == ("0b0", "0b1")

    # With a two bit selector choose one of four outputs, input passes through and unselected is always 0
    assert _dmux4way.evaluate(_in="0b0", sel2="0b00") == ("0b0", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b0", sel2="0b01") == ("0b0", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b0", sel2="0b10") == ("0b0", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b0", sel2="0b11") == ("0b0", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b1", sel2="0b00") == ("0b1", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b1", sel2="0b01") == ("0b0", "0b1", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b1", sel2="0b10") == ("0b0", "0b0", "0b1", "0b0")
    assert _dmux4way.evaluate(_in="0b1", sel2="0b11") == ("0b0", "0b0", "0b0", "0b1")
    
    # With a 3 bit selector choose one of 8 outputs, input passes through and unselected is always 0
    assert _dmux8way.evaluate(_in="0b0", sel3="0b000") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b0", sel3="0b001") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b0", sel3="0b010") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b0", sel3="0b011") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b0", sel3="0b100") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b0", sel3="0b101") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b0", sel3="0b110") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b0", sel3="0b111") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b000") == ("0b1", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b001") == ("0b0", "0b1", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b010") == ("0b0", "0b0", "0b1", "0b0", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b011") == ("0b0", "0b0", "0b0", "0b1", "0b0", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b100") == ("0b0", "0b0", "0b0", "0b0", "0b1", "0b0", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b101") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b1", "0b0", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b110") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b1", "0b0")
    assert _dmux8way.evaluate(_in="0b1", sel3="0b111") == ("0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b0", "0b1")

    # Computes the sum of 2 x 1 bit inputs, output carry bit & sum bit
    assert _halfAdder.evaluate(a="0b0", b="0b0") == ("0b0", "0b0")
    assert _halfAdder.evaluate(a="0b0", b="0b1") == ("0b0", "0b1")
    assert _halfAdder.evaluate(a="0b1", b="0b0") == ("0b0", "0b1")
    assert _halfAdder.evaluate(a="0b1", b="0b1") == ("0b1", "0b0")

    # Computes the sum of 3 x 1 bit inputs, output carry bit & sum bit
    assert _fullAdder.evaluate(a="0b0", b="0b0", c="0b0") == ("0b0", "0b0")
    assert _fullAdder.evaluate(a="0b1", b="0b1", c="0b1") == ("0b1", "0b1")
    assert _fullAdder.evaluate(a="0b1", b="0b0", c="0b0") == ("0b0", "0b1")
    assert _fullAdder.evaluate(a="0b0", b="0b1", c="0b0") == ("0b0", "0b1")
    assert _fullAdder.evaluate(a="0b0", b="0b0", c="0b1") == ("0b0", "0b1")
    assert _fullAdder.evaluate(a="0b0", b="0b1", c="0b1") == ("0b1", "0b0")
    assert _fullAdder.evaluate(a="0b1", b="0b0", c="0b1") == ("0b1", "0b0")
    assert _fullAdder.evaluate(a="0b1", b="0b1", c="0b0") == ("0b1", "0b0")

    # Adds two 16-bit values and output 16 bit result, the most significant carry bit is ignored
    assert _add16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000") == "0b0000000000000000"
    assert _add16.evaluate(a16="0b0000000000000001", b16="0b0000000000000001") == "0b0000000000000010"
    assert _add16.evaluate(a16="0b0000000000000001", b16="0b0000000000001111") == "0b0000000000010000"
    assert _add16.evaluate(a16="0b1111111111111110", b16="0b0000000000000001") == "0b1111111111111111"
    assert _add16.evaluate(a16="0b1111111100000000", b16="0b0000000000000000") == "0b1111111100000000"
    assert _add16.evaluate(a16="0b0000000011111111", b16="0b0000000000000000") == "0b0000000011111111"
    assert _add16.evaluate(a16="0b0000000000000000", b16="0b1111111100000000") == "0b1111111100000000"
    assert _add16.evaluate(a16="0b0000000000000000", b16="0b0000000011111111") == "0b0000000011111111"

    # Increment a 16 bit number
    assert _inc16.evaluate(_in16="0b0000000000000000") == "0b0000000000000001"
    assert _inc16.evaluate(_in16="0b0000000000000010") == "0b0000000000000011"
    assert _inc16.evaluate(_in16="0b0000000000000011") == "0b0000000000000100"
    assert _inc16.evaluate(_in16="0b1111111111111110") == "0b1111111111111111"

    # ALU: addition
    assert _alu.evaluate(x="0b0000000000000000", y="0b0000000000000000", zx="0b0", zy="0b0", nx="0b0", ny="0b0", f="0b1", no="0b0") == ("0b0000000000000000", "0b1", "0b0")
    assert _alu.evaluate(x="0b0000000000000001", y="0b0000000000000001", zx="0b0", zy="0b0", nx="0b0", ny="0b0", f="0b1", no="0b0") == ("0b0000000000000010", "0b0", "0b0")

    # ALU: zx/yx
    assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b1", zy="0b1", nx="0b0", ny="0b0", f="0b1", no="0b0") == ("0b0000000000000000", "0b1", "0b0")
    assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b1", zy="0b0", nx="0b0", ny="0b0", f="0b1", no="0b0") == ("0b1111111111111111", "0b0", "0b1")
    assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b0", zy="0b1", nx="0b0", ny="0b0", f="0b1", no="0b0") == ("0b1111111111111111", "0b0", "0b1")

    # ALU: nx/ny
    assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b0", zy="0b0", nx="0b1", ny="0b1", f="0b1", no="0b0") == ("0b0000000000000000", "0b1", "0b0")
    assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b0", zy="0b0", nx="0b1", ny="0b0", f="0b1", no="0b0") == ("0b1111111111111111", "0b0", "0b1")
    assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b0", zy="0b0", nx="0b0", ny="0b1", f="0b1", no="0b0") == ("0b1111111111111111", "0b0", "0b1")

    # ALU: and
    assert _alu.evaluate(x="0b0000000000000000", y="0b0000000000000000", zx="0b0", zy="0b0", nx="0b0", ny="0b0", f="0b0", no="0b0") == ("0b0000000000000000", "0b1", "0b0")
    assert _alu.evaluate(x="0b0000000000000000", y="0b1111111111111111", zx="0b0", zy="0b0", nx="0b0", ny="0b0", f="0b0", no="0b0") == ("0b0000000000000000", "0b1", "0b0")
    assert _alu.evaluate(x="0b1111111111111111", y="0b0000000000000000", zx="0b0", zy="0b0", nx="0b0", ny="0b0", f="0b0", no="0b0") == ("0b0000000000000000", "0b1", "0b0")
    assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b0", zy="0b0", nx="0b0", ny="0b0", f="0b0", no="0b0") == ("0b1111111111111111", "0b0", "0b1")

    # ALU: not(and)
    assert _alu.evaluate(x="0b1111111111111111", y="0b0000000000000000", zx="0b0", zy="0b0", nx="0b0", ny="0b0", f="0b0", no="0b1") == ("0b1111111111111111", "0b0", "0b1")
    assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b0", zy="0b0", nx="0b0", ny="0b0", f="0b0", no="0b1") == ("0b0000000000000000", "0b1", "0b0")

    # DFF
    assert _dff.evaluate(_in="0b0", load="0b0") == ("0b0", "0b1")  # Q=0 (initial)
    assert _dff.evaluate(_in="0b1", load="0b1") == ("0b1", "0b0")  # Q=1 (set 1)
    assert _dff.evaluate(_in="0b1", load="0b0") == ("0b1", "0b0")  # Q=1 (no change)
    assert _dff.evaluate(_in="0b0", load="0b0") == ("0b1", "0b0")  # Q=1 (no change)
    assert _dff.evaluate(_in="0b0", load="0b1") == ("0b0", "0b0")  # Q=0 (set 0 / reset)
    assert _dff.evaluate(_in="0b1", load="0b0") == ("0b0", "0b1")  # Q=1 (no change)
    assert _dff.evaluate(_in="0b0", load="0b0") == ("0b0", "0b1")  # Q=1 (no change)

    # # 1 bit register, if load emit in else previous value
    assert _bit.evaluate(_in="0b0", load="0b0") == "0b0"
    assert _bit.evaluate(_in="0b0", load="0b1") == "0b0"
    assert _bit.evaluate(_in="0b1", load="0b0") == "0b0"
    assert _bit.evaluate(_in="0b1", load="0b1") == "0b1"
    assert _bit.evaluate(_in="0b0", load="0b0") == "0b1"

    # # 16-bit register, if load emit in else previous value
    assert _register.evaluate(_in16="0b0000000000000000", load="0b0") == "0b0000000000000000"
    assert _register.evaluate(_in16="0b0000000000000000", load="0b1") == "0b0000000000000000"
    assert _register.evaluate(_in16="0b1111111111111111", load="0b0") == "0b0000000000000000"
    assert _register.evaluate(_in16="0b1111111111111111", load="0b1") == "0b1111111111111111"
    assert _register.evaluate(_in16="0b0000000000000001", load="0b1") == "0b0000000000000001"
    assert _register.evaluate(_in16="0b1000000000000000", load="0b1") == "0b1000000000000000"

    # # PC: load (inc=0, reset=0)
    assert _pc.evaluate(_in16="0b0000000000000000", load="0b0", inc="0b0", reset="0b0") == "0b0000000000000000"
    assert _pc.evaluate(_in16="0b1111111111111111", load="0b0", inc="0b0", reset="0b0") == "0b0000000000000000"
    assert _pc.evaluate(_in16="0b1111111111111111", load="0b1", inc="0b0", reset="0b0") == "0b1111111111111111"
    assert _pc.evaluate(_in16="0b0000000000000000", load="0b1", inc="0b0", reset="0b0") == "0b0000000000000000"
    assert _pc.evaluate(_in16="0b1000000000000000", load="0b1", inc="0b0", reset="0b0") == "0b1000000000000000"
    assert _pc.evaluate(_in16="0b0000000000000001", load="0b1", inc="0b0", reset="0b0") == "0b0000000000000001"

    # PC: inc/reset
    assert _pc.evaluate(_in16="0b0000000000000000", load="0b0", inc="0b1", reset="0b0") == "0b0000000000000010"
    assert _pc.evaluate(_in16="0b1111111111111111", load="0b0", inc="0b0", reset="0b1") == "0b0000000000000000"

    # PC: reset>load>inc
    assert _pc.evaluate(_in16="0b1111111111111111", load="0b1", inc="0b1", reset="0b1") == "0b0000000000000000"
    assert _pc.evaluate(_in16="0b0000000000000100", load="0b1", inc="0b1", reset="0b0") == "0b0000000000000100"

    # Memory of 8 registers, each 16 bit-wide
    assert _ram8.evaluate(_in16="0b0000000000000001", load="0b1", addr3="0b000") == "0b0000000000000001"
    assert _ram8.evaluate(_in16="0b1000000000000000", load="0b1", addr3="0b111") == "0b1000000000000000"
    assert _ram8.evaluate(_in16="0b0000000000000000", load="0b0", addr3="0b000") == "0b0000000000000001"
    assert _ram8.evaluate(_in16="0b0000000000000000", load="0b0", addr3="0b111") == "0b1000000000000000"

    # Memory of 64 registers, each 16 bit-wide
    assert _ram64.evaluate(_in16="0b0000000000000001", load="0b1", addr6="0b000000") == "0b0000000000000001"
    assert _ram64.evaluate(_in16="0b1000000000000000", load="0b1", addr6="0b000111") == "0b1000000000000000"
    assert _ram64.evaluate(_in16="0b0000000000000000", load="0b0", addr6="0b000000") == "0b0000000000000001"
    assert _ram64.evaluate(_in16="0b0000000000000000", load="0b0", addr6="0b000111") == "0b1000000000000000"


if __name__ == "__main__":
    main()
