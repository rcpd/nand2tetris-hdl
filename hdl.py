""""
Python implementation of the HACK architecture modelled after the Nand2Tetris HDL
NAND is a primitive implemented at the hardware level so need to define the logic ourselves
All subsequent gates can be expressed via increasingly complex abstractions of NAND

h = LSB ---------------------------------------------- MSB
h =  0  1  2  3  4  5  6  7  8   9  10  11  12  13  14  15
p = 15 14 13 12 11 10  9  8  7   6   5   4   3   2   1   0
p = -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16

"""

from datetime import datetime

# TODO: reduce multiple instantiations of classes where not required
# TODO: disable eval code


class Gate(object):
    def __init__(self):
        # Gates values should only be initialized at runtime
        self.a = None
        self.b = None
        self.c = None
        self._in = None
        self._in3 = None
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
        self.addr9 = None
        self.addr12 = None
        self.addr13 = None
        self.addr14 = None
        self.addr15 = None

    def evaluate(self, a=None, b=None, c=None, _in=None, _in3=None, sel=None, sel2=None, sel3=None, _in8=None,
                 _in16=None, a16=None, b16=None, c16=None, d16=None, e16=None, f16=None, g16=None, h16=None, x=None,
                 y=None, zx=None, nx=None, zy=None, ny=None, f=None, no=None, load=None, inc=None, reset=None,
                 addr3=None, addr6=None, addr9=None, addr12=None, addr13=None, addr14=None, addr15=None):
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

        if _in3 is not None:
            if type(_in3) is not str:
                _in3 = bin(_in3)
            if int(_in3, 2) < 0 or int(_in3, 2) > 7:
                raise RuntimeError("_in3 input must be 3 bits")
            self._in3 = _in3

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

        if addr9 is not None:
            if type(addr9) is not str:
                addr9 = bin(addr9)
            if int(addr9, 2) < 0 or int(addr9, 2) > 777:
                raise RuntimeError("addr9 input must be 9 bits")
            self.addr9 = addr9

        if addr12 is not None:
            if type(addr12) is not str:
                addr12 = bin(addr12)
            if int(addr12, 2) < 0 or int(addr12, 2) > 4095:
                raise RuntimeError("addr12 input must be 12 bits")
            self.addr12 = addr12

        if addr13 is not None:
            if type(addr13) is not str:
                addr13 = bin(addr13)
            if int(addr13, 2) < 0 or int(addr13, 2) > 8191:
                raise RuntimeError("addr13 input must be 13 bits")
            self.addr13 = addr13

        if addr14 is not None:
            if type(addr14) is not str:
                addr14 = bin(addr14)
            if int(addr14, 2) < 0 or int(addr14, 2) > 16383:
                raise RuntimeError("addr14 input must be 14 bits")
            self.addr14 = addr14

        if addr15 is not None:
            if type(addr15) is not str:
                addr15 = bin(addr15)
            if int(addr15, 2) < 0 or int(addr15, 2) > 32767:
                raise RuntimeError("addr15 input must be 15 bits")
            self.addr15 = addr15

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
            byte_str += NotGate().evaluate(_in="0b" + self._in16[i * -1])[2:]
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
            byte_str += AndGate().evaluate(a="0b" + self.a16[i * -1], b="0b" + self.b16[i * -1])[2:]
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
            byte_str += OrGate().evaluate(a="0b" + self.a16[i * -1], b="0b" + self.b16[i * -1])[2:]
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
        or0 = OrGate().evaluate(a="0b" + self._in8[-1], b="0b" + self._in8[-2])
        or1 = OrGate().evaluate(a="0b" + self._in8[-3], b="0b" + self._in8[-4])
        or2 = OrGate().evaluate(a="0b" + self._in8[-5], b="0b" + self._in8[-6])
        or3 = OrGate().evaluate(a="0b" + self._in8[-7], b="0b" + self._in8[-8])

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
            byte_str += Mux().evaluate(a="0b" + self.a16[i * -1], b="0b" + self.b16[i * -1], sel=self.sel)[2:]
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
        mux16_ab = Mux16().evaluate(a16=self.a16, b16=self.b16, sel="0b" + self.sel2[-1])
        mux16_cd = Mux16().evaluate(a16=self.c16, b16=self.d16, sel="0b" + self.sel2[-1])
        return Mux16().evaluate(a16=mux16_ab, b16=mux16_cd, sel="0b" + self.sel2[-2])


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
                                            sel2="0b" + self.sel3[-2] + self.sel3[-1])
        mux4way16_eh = Mux4Way16().evaluate(a16=self.e16, b16=self.f16, c16=self.g16, d16=self.h16,
                                            sel2="0b" + self.sel3[-2] + self.sel3[-1])
        return Mux16().evaluate(a16=mux4way16_ad, b16=mux4way16_eh, sel="0b" + self.sel3[-3])


class DMux(Gate):
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


class DMux4Way(Gate):
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
        dmux_0_a, dmux_0_b = DMux().evaluate(_in=self._in, sel="0b" + self.sel2[-2])
        dmux_1_a, dmux_1_b = DMux().evaluate(_in=dmux_0_a, sel="0b" + self.sel2[-1])
        dmux_2_a, dmux_2_b = DMux().evaluate(_in=dmux_0_b, sel="0b" + self.sel2[-1])
        return dmux_1_a, dmux_1_b, dmux_2_a, dmux_2_b


class DMux8Way(Gate):
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
        dmux_a, dmux_b = DMux().evaluate(_in=self._in, sel="0b" + self.sel3[-3])
        dmux4_0 = DMux4Way().evaluate(_in=dmux_a, sel2="0b" + self.sel3[-2] + self.sel3[-1])
        dmux4_1 = DMux4Way().evaluate(_in=dmux_b, sel2="0b" + self.sel3[-2] + self.sel3[-1])
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
        carry[-1], _sum[-1] = HalfAdder().evaluate(a="0b" + self.a16[-1], b="0b" + self.b16[-1])
        for i in range(2, 17):  # -2 to -16
            i = i * -1
            carry[i], _sum[i] = FullAdder().evaluate(a="0b" + self.a16[i], b="0b" + self.b16[i], c=carry[i + 1])

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

        # python can't represent simultaneous eval, this will always break and cause issues after n runs
        self.s_nor = NorGate().evaluate(a=s_and, b=self.r_nor)
        self.r_nor = NorGate().evaluate(a=self.s_nor, b=r_and)

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
        m_out = Mux().evaluate(a=self.d_out, b=self._in, sel=self.load)  # load=emit previous or new
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

    def __init__(self, name=None):
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
        self.d_out = "0b0000000000000000"
        self.name = name

    def calculate(self):
        # can't use range as Register has to save state
        self.bit0.evaluate(_in="0b" + self._in16[2], load=self.load)
        self.bit1.evaluate(_in="0b" + self._in16[3], load=self.load)
        self.bit2.evaluate(_in="0b" + self._in16[4], load=self.load)
        self.bit3.evaluate(_in="0b" + self._in16[5], load=self.load)
        self.bit4.evaluate(_in="0b" + self._in16[6], load=self.load)
        self.bit5.evaluate(_in="0b" + self._in16[7], load=self.load)
        self.bit6.evaluate(_in="0b" + self._in16[8], load=self.load)
        self.bit7.evaluate(_in="0b" + self._in16[9], load=self.load)
        self.bit8.evaluate(_in="0b" + self._in16[10], load=self.load)
        self.bit9.evaluate(_in="0b" + self._in16[11], load=self.load)
        self.bit10.evaluate(_in="0b" + self._in16[12], load=self.load)
        self.bit11.evaluate(_in="0b" + self._in16[13], load=self.load)
        self.bit12.evaluate(_in="0b" + self._in16[14], load=self.load)
        self.bit13.evaluate(_in="0b" + self._in16[15], load=self.load)
        self.bit14.evaluate(_in="0b" + self._in16[16], load=self.load)
        self.bit15.evaluate(_in="0b" + self._in16[17], load=self.load)
        self.d_out = "0b" + self.bit0.d_out[2:] + self.bit1.d_out[2:] + self.bit2.d_out[2:] + self.bit3.d_out[2:] \
                     + self.bit4.d_out[2:] + self.bit5.d_out[2:] + self.bit6.d_out[2:] + self.bit7.d_out[2:] \
                     + self.bit8.d_out[2:] + self.bit9.d_out[2:] + self.bit10.d_out[2:] + self.bit11.d_out[2:] \
                     + self.bit12.d_out[2:] + self.bit13.d_out[2:] + self.bit14.d_out[2:] + self.bit15.d_out[2:]
        return self.d_out


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

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.r0 = Register(name="r0")
        self.r1 = Register(name="r1")
        self.r2 = Register(name="r2")
        self.r3 = Register(name="r3")
        self.r4 = Register(name="r4")
        self.r5 = Register(name="r5")
        self.r6 = Register(name="r6")
        self.r7 = Register(name="r7")
        self.r0_out = "0b0000000000000000"
        self.r1_out = "0b0000000000000000"
        self.r2_out = "0b0000000000000000"
        self.r3_out = "0b0000000000000000"
        self.r4_out = "0b0000000000000000"
        self.r5_out = "0b0000000000000000"
        self.r6_out = "0b0000000000000000"
        self.r7_out = "0b0000000000000000"
        self.d_out = "0b0000000000000000"

    def calculate(self):
        # only evaluate selected Register (python performance optimisation)
        dmux8w = DMux8Way().evaluate(_in=self.load, sel3=self.addr3)
        if self.addr3 == "0b000":
            self.r0_out = self.r0.evaluate(_in16=self._in16, load=dmux8w[0])
        elif self.addr3 == "0b001":
            self.r1_out = self.r1.evaluate(_in16=self._in16, load=dmux8w[1])
        elif self.addr3 == "0b010":
            self.r2_out = self.r2.evaluate(_in16=self._in16, load=dmux8w[2])
        elif self.addr3 == "0b011":
            self.r3_out = self.r3.evaluate(_in16=self._in16, load=dmux8w[3])
        elif self.addr3 == "0b100":
            self.r4_out = self.r4.evaluate(_in16=self._in16, load=dmux8w[4])
        elif self.addr3 == "0b101":
            self.r5_out = self.r5.evaluate(_in16=self._in16, load=dmux8w[5])
        elif self.addr3 == "0b110":
            self.r6_out = self.r6.evaluate(_in16=self._in16, load=dmux8w[6])
        elif self.addr3 == "0b111":
            self.r7_out = self.r7.evaluate(_in16=self._in16, load=dmux8w[7])
        else:
            raise RuntimeError("Bad case in RAM8: %s" % "0b" + self.addr3)

        self.d_out = Mux8Way16().evaluate(a16=self.r0_out, b16=self.r1_out, c16=self.r2_out, d16=self.r3_out,
                                          e16=self.r4_out, f16=self.r5_out, g16=self.r6_out, h16=self.r7_out,
                                          sel3=self.addr3)
        return self.d_out


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

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.ram8_0 = RAM8(name="ram8_0")
        self.ram8_1 = RAM8(name="ram8_1")
        self.ram8_2 = RAM8(name="ram8_2")
        self.ram8_3 = RAM8(name="ram8_3")
        self.ram8_4 = RAM8(name="ram8_4")
        self.ram8_5 = RAM8(name="ram8_5")
        self.ram8_6 = RAM8(name="ram8_6")
        self.ram8_7 = RAM8(name="ram8_7")
        self.ram8_0_out = "0b0000000000000000"
        self.ram8_1_out = "0b0000000000000000"
        self.ram8_2_out = "0b0000000000000000"
        self.ram8_3_out = "0b0000000000000000"
        self.ram8_4_out = "0b0000000000000000"
        self.ram8_5_out = "0b0000000000000000"
        self.ram8_6_out = "0b0000000000000000"
        self.ram8_7_out = "0b0000000000000000"
        self.ram8_d_out = "0b0000000000000000"

    def calculate(self):
        # 3 MSB = RAM8 block, 3 LSB = Register
        # only evaluate selected RAM8 block (python performance optimisation)
        dmux8w = DMux8Way().evaluate(_in=self.load, sel3="0b" + self.addr6[-6:-3])
        if "0b" + self.addr6[-6:-3] == "0b000":
            self.ram8_0_out = self.ram8_0.evaluate(_in16=self._in16, load=dmux8w[0], addr3="0b" + self.addr6[-3:])
        elif "0b" + self.addr6[-6:-3] == "0b001":
            self.ram8_1_out = self.ram8_1.evaluate(_in16=self._in16, load=dmux8w[1], addr3="0b" + self.addr6[-3:])
        elif "0b" + self.addr6[-6:-3] == "0b010":
            self.ram8_2_out = self.ram8_2.evaluate(_in16=self._in16, load=dmux8w[2], addr3="0b" + self.addr6[-3:])
        elif "0b" + self.addr6[-6:-3] == "0b011":
            self.ram8_3_out = self.ram8_3.evaluate(_in16=self._in16, load=dmux8w[3], addr3="0b" + self.addr6[-3:])
        elif "0b" + self.addr6[-6:-3] == "0b100":
            self.ram8_4_out = self.ram8_4.evaluate(_in16=self._in16, load=dmux8w[4], addr3="0b" + self.addr6[-3:])
        elif "0b" + self.addr6[-6:-3] == "0b101":
            self.ram8_5_out = self.ram8_5.evaluate(_in16=self._in16, load=dmux8w[5], addr3="0b" + self.addr6[-3:])
        elif "0b" + self.addr6[-6:-3] == "0b110":
            self.ram8_6_out = self.ram8_6.evaluate(_in16=self._in16, load=dmux8w[6], addr3="0b" + self.addr6[-3:])
        elif "0b" + self.addr6[-6:-3] == "0b111":
            self.ram8_7_out = self.ram8_7.evaluate(_in16=self._in16, load=dmux8w[7], addr3="0b" + self.addr6[-3:])
        else:
            raise RuntimeError("Bad case in RAM64: %s" % "0b" + self.addr6[-6:-3])

        self.ram8_d_out = Mux8Way16().evaluate(
            a16=self.ram8_0_out, b16=self.ram8_1_out, c16=self.ram8_2_out, d16=self.ram8_3_out, e16=self.ram8_4_out,
            f16=self.ram8_5_out, g16=self.ram8_6_out, h16=self.ram8_7_out, sel3="0b" + self.addr6[-6:-3])

        return self.ram8_d_out


class RAM512(Gate):
    """
    Memory of 512 registers, each 16 bit-wide.
    Out holds the value stored at the memory location specified by address.
    If load==1, then the in value is loaded into the memory location specified by address

    CHIP RAM512 {
        IN in[16], load, address[9];
        OUT out[16];

    PARTS:
        DMux8Way(in=load, sel=address[0..2], a=r0, b=r1, c=r2, d=r3, e=r4, f=r5, g=r6, h=r7);
        RAM64(in=in, load=r0, address=address[3..8], out=r0out);
        RAM64(in=in, load=r1, address=address[3..8], out=r1out);
        RAM64(in=in, load=r2, address=address[3..8], out=r2out);
        RAM64(in=in, load=r3, address=address[3..8], out=r3out);
        RAM64(in=in, load=r4, address=address[3..8], out=r4out);
        RAM64(in=in, load=r5, address=address[3..8], out=r5out);
        RAM64(in=in, load=r6, address=address[3..8], out=r6out);
        RAM64(in=in, load=r7, address=address[3..8], out=r7out);
        Mux8Way16(a=r0out, b=r1out, c=r2out, d=r3out, e=r4out, f=r5out, g=r6out, h=r7out, sel=address[0..2], out=out);
    }
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.ram64_0 = RAM64(name="ram64_0")
        self.ram64_1 = RAM64(name="ram64_1")
        self.ram64_2 = RAM64(name="ram64_2")
        self.ram64_3 = RAM64(name="ram64_3")
        self.ram64_4 = RAM64(name="ram64_4")
        self.ram64_5 = RAM64(name="ram64_5")
        self.ram64_6 = RAM64(name="ram64_6")
        self.ram64_7 = RAM64(name="ram64_7")
        self.ram64_0_out = "0b0000000000000000"
        self.ram64_1_out = "0b0000000000000000"
        self.ram64_2_out = "0b0000000000000000"
        self.ram64_3_out = "0b0000000000000000"
        self.ram64_4_out = "0b0000000000000000"
        self.ram64_5_out = "0b0000000000000000"
        self.ram64_6_out = "0b0000000000000000"
        self.ram64_7_out = "0b0000000000000000"
        self.ram64_d_out = "0b0000000000000000"

    def calculate(self):
        # 3 MSB = RAM64 block, 6 LSB = RAM8>Register blocks
        # only evaluate selected RAM64 block (python performance optimisation)
        dmux8w = DMux8Way().evaluate(_in=self.load, sel3="0b" + self.addr9[-9:-6])
        if "0b" + self.addr9[-9:-6] == "0b000":
            self.ram64_0_out = self.ram64_0.evaluate(_in16=self._in16, load=dmux8w[0], addr6="0b" + self.addr9[-6:])
        elif "0b" + self.addr9[-9:-6] == "0b001":
            self.ram64_1_out = self.ram64_1.evaluate(_in16=self._in16, load=dmux8w[1], addr6="0b" + self.addr9[-6:])
        elif "0b" + self.addr9[-9:-6] == "0b010":
            self.ram64_2_out = self.ram64_2.evaluate(_in16=self._in16, load=dmux8w[2], addr6="0b" + self.addr9[-6:])
        elif "0b" + self.addr9[-9:-6] == "0b011":
            self.ram64_3_out = self.ram64_3.evaluate(_in16=self._in16, load=dmux8w[3], addr6="0b" + self.addr9[-6:])
        elif "0b" + self.addr9[-9:-6] == "0b100":
            self.ram64_4_out = self.ram64_4.evaluate(_in16=self._in16, load=dmux8w[4], addr6="0b" + self.addr9[-6:])
        elif "0b" + self.addr9[-9:-6] == "0b101":
            self.ram64_5_out = self.ram64_5.evaluate(_in16=self._in16, load=dmux8w[5], addr6="0b" + self.addr9[-6:])
        elif "0b" + self.addr9[-9:-6] == "0b110":
            self.ram64_6_out = self.ram64_6.evaluate(_in16=self._in16, load=dmux8w[6], addr6="0b" + self.addr9[-6:])
        elif "0b" + self.addr9[-9:-6] == "0b111":
            self.ram64_7_out = self.ram64_7.evaluate(_in16=self._in16, load=dmux8w[7], addr6="0b" + self.addr9[-6:])
        else:
            raise RuntimeError("Bad case in RAM512: %s" % "0b" + self.addr6[-9:-6])

        self.ram64_d_out = Mux8Way16().evaluate(
            a16=self.ram64_0_out, b16=self.ram64_1_out, c16=self.ram64_2_out, d16=self.ram64_3_out,
            e16=self.ram64_4_out, f16=self.ram64_5_out, g16=self.ram64_6_out, h16=self.ram64_7_out,
            sel3="0b" + self.addr9[-9:-6])

        return self.ram64_d_out


class RAM4K(Gate):
    """
    Memory of 4k registers, each 16 bit-wide.
    Out holds the value stored at the memory location specified by address.
    If load==1, then the in value is loaded into the memory location specified by address

    CHIP RAM4K {
        IN in[16], load, address[12];
        OUT out[16];
    
    PARTS:
        DMux8Way(in=load, sel=address[0..2], a=r0, b=r1, c=r2, d=r3, e=r4, f=r5, g=r6, h=r7);
        RAM512(in=in, load=r0, address=address[3..11], out=r0out);
        RAM512(in=in, load=r1, address=address[3..11], out=r1out);
        RAM512(in=in, load=r2, address=address[3..11], out=r2out);
        RAM512(in=in, load=r3, address=address[3..11], out=r3out);
        RAM512(in=in, load=r4, address=address[3..11], out=r4out);
        RAM512(in=in, load=r5, address=address[3..11], out=r5out);
        RAM512(in=in, load=r6, address=address[3..11], out=r6out);
        RAM512(in=in, load=r7, address=address[3..11], out=r7out);
        Mux8Way16(a=r0out, b=r1out, c=r2out, d=r3out, e=r4out, f=r5out, g=r6out, h=r7out, sel=address[0..2], out=out);
    }
    """
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.ram512_0 = RAM512(name="ram512_0")
        self.ram512_1 = RAM512(name="ram512_1")
        self.ram512_2 = RAM512(name="ram512_2")
        self.ram512_3 = RAM512(name="ram512_3")
        self.ram512_4 = RAM512(name="ram512_4")
        self.ram512_5 = RAM512(name="ram512_5")
        self.ram512_6 = RAM512(name="ram512_6")
        self.ram512_7 = RAM512(name="ram512_7")
        self.ram512_0_out = "0b0000000000000000"
        self.ram512_1_out = "0b0000000000000000"
        self.ram512_2_out = "0b0000000000000000"
        self.ram512_3_out = "0b0000000000000000"
        self.ram512_4_out = "0b0000000000000000"
        self.ram512_5_out = "0b0000000000000000"
        self.ram512_6_out = "0b0000000000000000"
        self.ram512_7_out = "0b0000000000000000"
        self.ram512_d_out = "0b0000000000000000"

    def calculate(self):
        # 3 MSB = RAM512 block, 9 LSB = RAM64>RAM8>Register blocks
        # only evaluate selected RAM512 block (python performance optimisation)
        dmux8w = DMux8Way().evaluate(_in=self.load, sel3="0b" + self.addr12[-12:-9])
        if "0b" + self.addr12[-12:-9] == "0b000":
            self.ram512_0_out = self.ram512_0.evaluate(_in16=self._in16, load=dmux8w[0], addr9=self.addr12[-9:])
        elif "0b" + self.addr12[-12:-9] == "0b001":
            self.ram512_1_out = self.ram512_1.evaluate(_in16=self._in16, load=dmux8w[1], addr9=self.addr12[-9:])
        elif "0b" + self.addr12[-12:-9] == "0b010":
            self.ram512_2_out = self.ram512_2.evaluate(_in16=self._in16, load=dmux8w[2], addr9=self.addr12[-9:])
        elif "0b" + self.addr12[-12:-9] == "0b011":
            self.ram512_3_out = self.ram512_3.evaluate(_in16=self._in16, load=dmux8w[3], addr9=self.addr12[-9:])
        elif "0b" + self.addr12[-12:-9] == "0b100":
            self.ram512_4_out = self.ram512_4.evaluate(_in16=self._in16, load=dmux8w[4], addr9=self.addr12[-9:])
        elif "0b" + self.addr12[-12:-9] == "0b101":
            self.ram512_5_out = self.ram512_5.evaluate(_in16=self._in16, load=dmux8w[5], addr9=self.addr12[-9:])
        elif "0b" + self.addr12[-12:-9] == "0b110":
            self.ram512_6_out = self.ram512_6.evaluate(_in16=self._in16, load=dmux8w[6], addr9=self.addr12[-9:])
        elif "0b" + self.addr12[-12:-9] == "0b111":
            self.ram512_7_out = self.ram512_7.evaluate(_in16=self._in16, load=dmux8w[7], addr9=self.addr12[-9:])
        else:
            raise RuntimeError("Bad case in RAM4K: %s" % "0b" + self.addr12[-12:-9])

        self.ram512_d_out = Mux8Way16().evaluate(
            a16=self.ram512_0_out, b16=self.ram512_1_out, c16=self.ram512_2_out, d16=self.ram512_3_out,
            e16=self.ram512_4_out, f16=self.ram512_5_out, g16=self.ram512_6_out, h16=self.ram512_7_out,
            sel3="0b" + self.addr12[-12:-9])

        return self.ram512_d_out


class RAM16K(Gate):
    """
    Memory of 16k registers, each 16 bit-wide.
    Out holds the value stored at the memory location specified by address.
    If load==1, then the in value is loaded into the memory location specified by address

    CHIP RAM16K {
        IN in[16], load, address[14];
        OUT out[16];
    
    PARTS:
        DMux4Way(in=load, sel=address[0..1], a=r0, b=r1, c=r2, d=r3);
        RAM4K(in=in, load=r0, address=address[2..13], out=r0out);
        RAM4K(in=in, load=r1, address=address[2..13], out=r1out);
        RAM4K(in=in, load=r2, address=address[2..13], out=r2out);
        RAM4K(in=in, load=r3, address=address[2..13], out=r3out);
        Mux4Way16(a=r0out, b=r1out, c=r2out, d=r3out, sel=address[0..1], out=out);
    }
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.ram4k_0 = RAM4K(name="ram4k_0")
        self.ram4k_1 = RAM4K(name="ram4k_1")
        self.ram4k_2 = RAM4K(name="ram4k_2")
        self.ram4k_3 = RAM4K(name="ram4k_3")
        self.ram4k_0_out = "0b0000000000000000"
        self.ram4k_1_out = "0b0000000000000000"
        self.ram4k_2_out = "0b0000000000000000"
        self.ram4k_3_out = "0b0000000000000000"
        self.ram4k_d_out = "0b0000000000000000"

    def calculate(self):
        # 2 MSB = RAM4K block, 12 LSB = RAM512>RAM64>RAM8>Register blocks
        # only evaluate selected RAM512 block (python performance optimisation)
        dmux4w = DMux4Way().evaluate(_in=self.load, sel2="0b" + self.addr14[-14:-12])
        if "0b" + self.addr14[-14:-12] == "0b00":
            self.ram4k_0_out = self.ram4k_0.evaluate(_in16=self._in16, load=dmux4w[0], addr12=self.addr14[-12:])
        elif "0b" + self.addr14[-14:-12] == "0b01":
            self.ram4k_1_out = self.ram4k_1.evaluate(_in16=self._in16, load=dmux4w[1], addr12=self.addr14[-12:])
        elif "0b" + self.addr14[-14:-12] == "0b10":
            self.ram4k_2_out = self.ram4k_2.evaluate(_in16=self._in16, load=dmux4w[2], addr12=self.addr14[-12:])
        elif "0b" + self.addr14[-14:-12] == "0b11":
            self.ram4k_3_out = self.ram4k_3.evaluate(_in16=self._in16, load=dmux4w[3], addr12=self.addr14[-12:])
        else:
            raise RuntimeError("Bad case in RAM16K: %s" % "0b" + self.addr12[-14:-12])

        self.ram4k_d_out = Mux4Way16().evaluate(
            a16=self.ram4k_0_out, b16=self.ram4k_1_out, c16=self.ram4k_2_out, d16=self.ram4k_3_out,
            sel2="0b" + self.addr14[-14:-12])

        return self.ram4k_d_out


class DMux3(Gate):
    """
    3 bit input, 1 bit select, 2 x 3 bit output dmux
    CHIP DMux3 {
        IN in[3], sel;
        OUT a[3], b[3];

    PARTS:
        //dmux1
        Nand(a=sel, b=sel, out=aNand1);
        Nand(a=in[0], b=aNand1, out=bNand1);
        Nand(a=sel, b=in[0], out=cNand1);
        Nand(a=bNand1, b=bNand1, out=a[0]);
        Nand(a=cNand1, b=cNand1, out=b[0]);
        //dmux2
        Nand(a=sel, b=sel, out=aNand2);
        Nand(a=in[1], b=aNand2, out=bNand2);
        Nand(a=sel, b=in[1], out=cNand2);
        Nand(a=bNand2, b=bNand2, out=a[1]);
        Nand(a=cNand2, b=cNand2, out=b[1]);
        //dmux3
        Nand(a=sel, b=sel, out=aNand3);
        Nand(a=in[2], b=aNand3, out=bNand3);
        Nand(a=sel, b=in[2], out=cNand3);
        Nand(a=bNand3, b=bNand3, out=a[2]);
        Nand(a=cNand3, b=cNand3, out=b[2]);
    }
    """

    def calculate(self):
        # dmux1
        nand_a0 = NandGate().evaluate(a=self.sel, b=self.sel)
        nand_b0 = NandGate().evaluate(a="0b" + self._in3[-1], b=nand_a0)
        nand_c0 = NandGate().evaluate(a=self.sel, b="0b" + self._in3[-1])
        a0 = NandGate().evaluate(a=nand_b0, b=nand_b0)[-1]
        b0 = NandGate().evaluate(a=nand_c0, b=nand_c0)[-1]

        # dmux2
        nand_b1 = NandGate().evaluate(a="0b" + self._in3[-2], b=nand_a0)
        nand_c1 = NandGate().evaluate(a=self.sel, b="0b" + self._in3[-2])
        a1 = NandGate().evaluate(a=nand_b1, b=nand_b1)[-1]
        b1 = NandGate().evaluate(a=nand_c1, b=nand_c1)[-1]

        # dmux3
        nand_b2 = NandGate().evaluate(a="0b" + self._in3[-3], b=nand_a0)
        nand_c2 = NandGate().evaluate(a=self.sel, b="0b" + self._in3[-3])
        a2 = NandGate().evaluate(a=nand_b2, b=nand_b2)[-1]
        b2 = NandGate().evaluate(a=nand_c2, b=nand_c2)[-1]

        return ("0b" + a2 + a1 + a0), ("0b" + b2 + b1 + b0)


class Dmux4Way3(Gate):
    """
    2 x 3 bit inputs, 2 bit select, 4 x 3 bit outputs
    CHIP DMux4Way3 {
        IN in[3], sel[2];
        OUT a[3], b[3], c[3], d[3];

    PARTS:
        DMux3(in=in, sel=sel[1], a=dIn0, b=dIn1);
        DMux3(in=dIn0, sel=sel[0], a=a, b=b);
        DMux3(in=dIn1, sel=sel[0], a=c, b=d);
    }
    """

    def calculate(self):
        dmux3_0a, dmux3_0b = DMux3().evaluate(_in3=self._in3, sel="0b" + self.sel2[-2])
        dmux3_1a, dmux3_1b = DMux3().evaluate(_in3=dmux3_0a, sel="0b" + self.sel2[-1])
        dmux3_2a, dmux3_2b = DMux3().evaluate(_in3=dmux3_0b, sel="0b" + self.sel2[-1])
        return dmux3_1a, dmux3_1b, dmux3_2a, dmux3_2b


class Screen(Gate):
    """
    Simulate a 256x512 monochrome screen where each row in the visual screen
    is represented by 32 consecutive 16-bit words.
    // No HDL, implemented in Java on the course
    // Screen(in=in[16],load=load,address=address[13],out=out[16]);
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.ram4k_0 = RAM4K(name="screen_ram4k_0")
        self.ram4k_1 = RAM4K(name="screen_ram4k_1")
        self.ram4k_0_out = "0b0000000000000000"
        self.ram4k_1_out = "0b0000000000000000"
        self.screen_out = "0b0000000000000000"

    def calculate(self):
        # MSB = RAM4K selector
        # only evaluate selected RAM4K block (python performance optimisation)
        dmux = DMux().evaluate(_in=self.load, sel="0b" + self.addr13[-13])

        if "0b" + self.addr13[-13] == "0b0":
            self.ram4k_0_out = self.ram4k_0.evaluate(_in16=self._in16, load=dmux[0], addr12=self.addr13[-12:])
        elif "0b" + self.addr13[-13] == "0b1":
            self.ram4k_1_out = self.ram4k_1.evaluate(_in16=self._in16, load=dmux[1], addr12=self.addr13[-12:])
        else:
            raise RuntimeError("Bad case in Screen: %s" % "0b" + self.addr13[-13])

        self.screen_out = Mux16().evaluate(a16=self.ram4k_0_out, b16=self.ram4k_1_out, sel="0b" + self.addr13[-13])

        return self.screen_out


class Memory(Gate):
    """
    16K+8K+1 memory block for RAM, Screen, Keyboard address ranges respectively
    CHIP Memory {
        IN in[16], load, address[15]; //0=LSB,14=MSB
        OUT out[16];

    PARTS:
        // determine which chip is being addressed from 2xMSB in address
        // A/B = RAM, C = SCREEN, D = KEYBOARD
        DMux4Way(in=true,sel[0]=address[13],sel[1]=address[14],a=aOut,b=bOut,c=cOut);
        Or(a=aOut,b=bOut,out=abOut);
    
        // determine what chip, if any, will load
        And(a=abOut,b=load,out=ramLoad); 
        And(a=cOut,b=load,out=screenLoad);
        
        // process memory maps: selective load, always read
        RAM16K(in=in,load=ramLoad,address=address[0..13],out=ramOut);
        Screen(in=in,load=screenLoad,address=address[0..12],out=screenOut); 
        Keyboard(out=keyOut);
        
        // select which out gets expressed
        Mux4Way16(a=ramOut,b=ramOut,c=screenOut,d=keyOut,sel[0]=address[13],sel[1]=address[14],out=out);
    }
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.ram16k = RAM16K(name="memory_ram16k")
        self.screen = Screen(name="memory_screen")
        self.keyboard = Register(name="memory_keyboard")
        self.ram16k_out = "0b0000000000000000"
        self.screen_out = "0b0000000000000000"
        self.keyboard_out = "0b0000000000000000"

    def calculate(self):
        # determine which chip is being addressed from 2xMSB in address
        dmux4w_a, dmux4w_b, dmux4w_c, dmux4w_d = DMux4Way().evaluate(
            _in="0b1", sel2="0b" + self.addr15[-15:-13])
        or0 = OrGate().evaluate(a=dmux4w_a, b=dmux4w_b)

        # determine what chip, if any, will load
        and0 = AndGate().evaluate(a=or0, b=self.load)
        and1 = AndGate().evaluate(a=dmux4w_c, b=self.load)

        # process memory maps: selective load, always read
        # only evaluate selected block (python performance optimisation)
        if "0b" + self.addr15[-15:-13] in ("0b00", "0b01"):
            self.ram16k_out = self.ram16k.evaluate(_in16=self._in16, load=and0, addr14="0b" + self.addr15[-14:])
        elif "0b" + self.addr15[-15:-13] == "0b10":
            self.screen_out = self.screen.evaluate(_in16=self._in16, load=and1, addr13="0b" + self.addr15[-13:])
        elif "0b" + self.addr15[-15:-13] == "0b11":
            self.keyboard_out = self.keyboard.evaluate(_in16=self._in16, load=self.load)
        else:
            raise RuntimeError("Bad case in Memory: %s" % "0b" + self.addr15[-15:-13])

        # select which out gets expressed
        return Mux4Way16().evaluate(a16=self.ram16k_out, b16=self.ram16k_out, c16=self.screen_out,
                                    d16=self.keyboard_out, sel2="0b" + self.addr15[-15:-13])


class ROM32K(Gate):
    """
    Read-Only memory (ROM) of 16K registers
    Implemented as RAM chips where load defaults to off
    // No HDL, implemented in Java on the course
    // ROM32K(address=address[15],out=out[16]);
    }
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.rom_load = "0b0"
        self.rom16k_0 = RAM16K(name="rom_ram16k_0")
        self.rom16k_1 = RAM16K(name="rom_ram16k_1")
        self.rom16k_0_out = "0b0000000000000000"
        self.rom16k_1_out = "0b0000000000000000"

    def calculate(self):
        # determine which chip is being addressed from MSB in address
        # process memory maps: selective load, always read
        # only evaluate selected block (python performance optimisation)
        if "0b" + self.addr15[-15] == "0b0":
            self.rom16k_0_out = self.rom16k_0.evaluate(_in16=self._in16, load=self.rom_load,
                                                       addr14="0b" + self.addr15[-14:])
        elif "0b" + self.addr15[-15] == "0b1":
            self.rom16k_1_out = self.rom16k_1.evaluate(_in16=self._in16, load=self.rom_load,
                                                       addr14="0b" + self.addr15[-14:])
        else:
            raise RuntimeError("Bad case in ROM32K: %s" % "0b" + self.addr15[-15])

        # select which out gets expressed
        return Mux16().evaluate(a16=self.rom16k_0_out, b16=self.rom16k_1_out, sel="0b" + self.addr15[-15])


class CPU(Gate):
    """
    CHIP CPU {
        IN  inM[16],         // M value input  (M = contents of RAM[A])
            instruction[16], // Instruction for execution
            reset;           // Signals whether to re-start the current
                             // program (reset==1) or continue executing
                             // the current program (reset==0).

        OUT outM[16],        // M value output
            writeM,          // Write to M?
            addressM[15],    // Address in data memory (of M)
            pc[15];          // address of next instruction

    PARTS:
        // var for opcode
        Not(in=instruction[15],out=notOpcode);
        Not(in=notOpcode,out=opcode);

        // Determine whether instruction is A or C type
        XNor(a=instruction[15],b=false,out=aType);

        // Solve whether writeM is false (A inst) or variable (C inst)
        XNor(a=aType,b=false,out=aTypeXNor);
        And(a=aTypeXNor,b=instruction[3],out=writeM);

        // emit address (A inst) or result (C inst)
        Mux16(a=ALUout,b=instruction,sel=notOpcode,out=mux1out);

        // Solve whether aRegisterLoad is true (A inst) or variable (C inst)
        Or(a=aType,b=instruction[5],out=aRegisterLoad);

        // Solve whether dRegisterLoad is false (A inst) or variable (C inst)
        And(a=aTypeXNor,b=instruction[4],out=dRegisterLoad);

        // addressM: emit current or previous address based on load
        // dRegisterOut: emit current or previous result based on load
        // mux2out: emit (current/previous address) or (original value stored at address)
        ARegister(in=mux1out,load=aRegisterLoad,out=aRegisterOut,out[0..14]=addressM);
        DRegister(in=ALUout,load=dRegisterLoad,out=dRegisterOut);
        Mux16(a=aRegisterOut,b=inM,sel=instruction[12],out=mux2out);

        // evaluate jump code
        // block 1: evaluate jmp bits for 111 or other (removed)

        // block 2: evaluate zr/ng bits
        DMux4Way3(in=instruction[0..2],
        sel[0]=ngOut,sel[1]=zrOut,
        a[0]=aOut0,a[1]=aOut1,a[2]=aOut2,
        b[0]=bOut0,b[1]=bOut1,b[2]=bOut2,
        c[0]=cOut0,c[1]=cOut1,c[2]=cOut2,
        d[0]=dOut0,d[1]=dOut1,d[2]=dOut2);

        // block 2-1: evaluate zr=0/ng=0 (011,001,101,111 == LSB=1)
        And(a=aOut0,b=true,out=out21);

        // block 2-2: evaluate zr=0/ng=1 (100,101,110,111 == MSB=1)
        And(a=true,b=bOut2,out=out22);

        // block 2-3: evaluate zr=1/ng=0 (010,011,110,111 == MidB=1)
        And(a=true,b=cOut1,out=out23);

        // block 2-4: evaluate zr=1/ng=1 (should never happen)

        // block 3: evaluating block 1/2 outputs
        Or(a=out21,b=out22,out=out2122);
        Or(a=out22,b=out23,out=out2223);
        Or(a=out2122,b=out2223,out=jumpOut);

        // Solve whether jumpOut is false (A inst) or variable (C inst)
        And(a=aTypeXNor,b=jumpOut,out=jumpOutFinal);

        // Back to CPU glue code
        Not(in=jumpOutFinal,out=notJumpOutFinal);
        PC(in=aRegisterOut,load=jumpOutFinal,inc=notJumpOutFinal,reset=reset,out[0..14]=pc);
        ALU(x=dRegisterOut,y=mux2out,zx=instruction[11],nx=instruction[10],zy=instruction[9],ny=instruction[8],
            f=instruction[7],no=instruction[6],out=outM,out=ALUout,zr=zrOut,ng=ngOut);
    }
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.a_register = Register(name="cpu_a_register")
        self.d_register = Register(name="cpu_d_register")
        self.ALU = ALU()
        self.PC = PC()
        self.ALU.zr = "0b1"
        self.ALU.ng = "0b0"
        self.a_out = "0b0000000000000000"
        self.d_out = "0b0000000000000000"
        self.m_out = "0b0000000000000000"
        self.pc_out = "0b0000000000000000"
        self.write_out = "0b0"

    def calculate(self):
        # opcode var
        not_opcode = NotGate().evaluate(_in="0b" + self._in16[-16])

        # Determine whether instruction is A or C type
        a_type = XNorGate().evaluate(a="0b" + self._in16[-16], b="0b0")

        # Solve whether writeM is false (A inst) or variable (C inst)
        a_type_xnor = XNorGate().evaluate(a=a_type, b="0b0")
        self.write_out = AndGate().evaluate(a=a_type_xnor, b="0b" + self._in16[-4])  # M dest

        # emit address (A inst) or result (C inst)
        mux1out = Mux16().evaluate(a16=self.m_out, b16=self._in16, sel=not_opcode)

        # Solve whether aRegisterLoad is true (A inst) or variable (C inst)
        a_load = OrGate().evaluate(a=a_type, b="0b" + self._in16[-6])  # A dest

        # Solve whether dRegisterLoad is false (A inst) or variable (C inst)
        d_load = AndGate().evaluate(a=a_type_xnor, b="0b" + self._in16[-5])  # D dest

        # mux2out: emit (current/previous address) or (original value stored at address)
        self.a_out = self.a_register.evaluate(_in16=mux1out, load=a_load)
        self.d_out = self.d_register.evaluate(_in16=self.m_out, load=d_load)
        mux2out = Mux16().evaluate(a16=self.a_out, b16=self.b16, sel="0b" + self._in16[-13])  # sel=A/M bit

        # pass comp bits to ALU
        self.m_out = self.ALU.evaluate(
            x=self.d_out, y=mux2out, zx="0b" + self._in16[-12], nx="0b" + self._in16[-11], zy="0b" + self._in16[-10],
            ny="0b" + self._in16[-9], f="0b" + self._in16[-8], no="0b" + self._in16[-7])[0]  # comp bits

        # evaluate jump code
        a_out, b_out, c_out, d_out = Dmux4Way3().evaluate(
            _in3="0b" + self._in16[-3:], sel2="0b" + self.ALU.zr[-1] + self.ALU.ng[-1])  # JMP bits
        out21 = AndGate().evaluate(a="0b" + a_out[-1], b="0b1")
        out22 = AndGate().evaluate(a="0b1", b="0b" + b_out[-3])
        out23 = AndGate().evaluate(a="0b1", b="0b" + c_out[-2])
        out2122 = OrGate().evaluate(a=out21, b=out22)
        out2223 = OrGate().evaluate(a=out22, b=out23)
        jump_out = OrGate().evaluate(a=out2122, b=out2223)

        # solve whether jumpOut is false (A inst) or variable (C inst)
        jump_out_final = AndGate().evaluate(a=a_type_xnor, b=jump_out)

        # jump or increment
        not_jump_out_final = NotGate().evaluate(_in=jump_out_final)
        self.pc_out = self.PC.evaluate(_in16=self.a_out, load=jump_out_final, inc=not_jump_out_final, reset=self.reset)

        # python only changes due to timing difference in DFF implementation
        # retrieve d_out from current cycle for unit test
        d_out = self.d_register.evaluate(_in16=self.m_out, load=d_load)

        # retrieve a_out from current cycle for unit test
        mux1out = Mux16().evaluate(a16=self.m_out, b16=self._in16, sel=not_opcode)
        a_out = self.a_register.evaluate(_in16=mux1out, load=a_load)

        # retrieve m_out from current cycle when "A" in destination and "M" is not
        # (ignore ALU result and pass through inM)
        nand_m = NandGate().evaluate(a="0b" + self._in16[-4], b="0b1")  # M dest
        and_ac = AndGate().evaluate(a="0b" + self._in16[-6], b="0b" + self._in16[-16])  # A dest, C inst
        and_ac_not_m = AndGate().evaluate(a=nand_m, b=and_ac)
        m_out = Mux16().evaluate(a16=self.m_out, b16=self.b16, sel=and_ac_not_m)

        return m_out, self.write_out, a_out, self.pc_out, d_out


class Computer(Gate):
    """
    CHIP Computer {

    IN reset;

    PARTS:
        ROM32K(address=pcOut,out=romOut);
        CPU(inM=ramOut,instruction=romOut,reset=reset,outM=ramData,writeM=writeMem,addressM=addressRAM,pc=pcOut);
        Memory(in=ramData,load=writeMem,address=addressRAM,out=ramOut);
    }
    """
    def __init__(self, name=None, debug=False):
        super().__init__()
        self.name = name
        self.debug = debug
        self.CPU = CPU()
        self.ROM32K = ROM32K()
        self.Memory = Memory()
        self.write_m = "0b0"
        self._reset = "0b0"
        self.pc_out = "0b0000000000000000"
        self.rom_out = "0b0000000000000000"
        self.a_out = "0b0000000000000000"
        self.d_out = "0b0000000000000000"
        self.m_out = "0b0000000000000000"

    def flash_rom(self, program):
        self.ROM32K.rom_load = "0b1"

        for i, command in enumerate(program):
            if self.debug:
                print(format(i, '#017b'), "0b" + command.strip(),
                      self.ROM32K.evaluate(_in16="0b" + command.strip(),
                                           addr15=format(i, '#017b')))  # bin(i) & pad to 16 bit
            else:
                self.ROM32K.evaluate(_in16="0b" + command.strip(), load="0b1", addr15=format(i, '#017b'))

        self.ROM32K.rom_load = "0b0"
        if self.debug:
            print()

    def calculate(self):
        self.rom_out = self.ROM32K.evaluate(_in16=self.m_out, addr15="0b" + self.pc_out[-15:])
        self.m_out, self.write_m, self.a_out, self.pc_out, self.d_out = \
            self.CPU.evaluate(_in16=self.rom_out, b16=self.m_out, reset=self._reset)
        self.m_out = self.Memory.evaluate(_in16=self.m_out, load=self.write_m, addr15="0b" + self.a_out[-15:])

        if self.debug:
            print(self.rom_out, self.a_out, self.m_out, self.write_m, self.pc_out, self.d_out)

        return self.rom_out, self.a_out, self.m_out, self.write_m, self.pc_out, self.d_out


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


def main(test_all=False, debug=False):
    """
    Sanity check our truth tables for each gate as implemented
    """
    if test_all:
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
        _dmux3 = DMux3()
        _dmux4way = DMux4Way()
        _dmux4way3 = Dmux4Way3()
        _dmux8way = DMux8Way()
        _halfAdder = HalfAdder()
        _fullAdder = FullAdder()
        _add16 = Add16()
        _inc16 = Inc16()
        _alu = ALU()
        _dff = DFF()
        _bit = Bit()
        _register = Register(name="register_assert")
        _pc = PC()
        _ram8 = RAM8(name="ram8_assert")
        _ram64 = RAM64(name="ram64_assert")
        _ram512 = RAM512(name="ram512_assert")
        _ram4k = RAM4K(name="ram4k_assert")
        _ram16k = RAM16K(name="ram16k_assert")
        _screen = Screen(name="screen_assert")
        _cpu = CPU(name="cpu_assert")
        _memory = Memory(name="memory_assert")
        _rom32k = ROM32K(name="rom32k_assert")

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
        assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", sel2="0b00") == "0b0000000000000000"
        assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", sel2="0b01") == "0b0000000000000000"
        assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", sel2="0b10") == "0b0000000000000000"
        assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", sel2="0b11") == "0b0000000000000000"
        assert _mux4way16.evaluate(a16="0b1111111111111111", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", sel2="0b00") == "0b1111111111111111"
        assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b1111111111111111", c16="0b0000000000000000",
                                   d16="0b0000000000000000", sel2="0b01") == "0b1111111111111111"
        assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b1111111111111111",
                                   d16="0b0000000000000000", sel2="0b10") == "0b1111111111111111"
        assert _mux4way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b1111111111111111", sel2="0b11") == "0b1111111111111111"

        # Mux16 but with 8 x 16 bit inputs, one 16 bit output, 3 bit selector, only selected is emitted
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000",
                                   g16="0b0000000000000000", h16="0b0000000000000000",
                                   sel3="0b000") == "0b0000000000000000"
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000",
                                   g16="0b0000000000000000", h16="0b0000000000000000",
                                   sel3="0b001") == "0b0000000000000000"
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000",
                                   g16="0b0000000000000000", h16="0b0000000000000000",
                                   sel3="0b010") == "0b0000000000000000"
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000",
                                   g16="0b0000000000000000", h16="0b0000000000000000",
                                   sel3="0b011") == "0b0000000000000000"
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000",
                                   g16="0b0000000000000000", h16="0b0000000000000000",
                                   sel3="0b100") == "0b0000000000000000"
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000",
                                   g16="0b0000000000000000", h16="0b0000000000000000",
                                   sel3="0b101") == "0b0000000000000000"
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000",
                                   g16="0b0000000000000000", h16="0b0000000000000000",
                                   sel3="0b110") == "0b0000000000000000"
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000",
                                   g16="0b0000000000000000", h16="0b0000000000000000",
                                   sel3="0b111") == "0b0000000000000000"
        assert _mux8way16.evaluate(a16="0b1111111111111111", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000",
                                   g16="0b0000000000000000", h16="0b0000000000000000",
                                   sel3="0b000") == "0b1111111111111111"
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b1111111111111111", c16="0b0000000000000000",
                                   d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000",
                                   g16="0b0000000000000000", h16="0b0000000000000000",
                                   sel3="0b001") == "0b1111111111111111"
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b1111111111111111",
                                   d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000",
                                   g16="0b0000000000000000", h16="0b0000000000000000",
                                   sel3="0b010") == "0b1111111111111111"
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b1111111111111111", e16="0b0000000000000000", f16="0b0000000000000000",
                                   g16="0b0000000000000000", h16="0b0000000000000000",
                                   sel3="0b011") == "0b1111111111111111"
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", e16="0b1111111111111111", f16="0b0000000000000000",
                                   g16="0b0000000000000000", h16="0b0000000000000000",
                                   sel3="0b100") == "0b1111111111111111"
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", e16="0b0000000000000000", f16="0b1111111111111111",
                                   g16="0b0000000000000000", h16="0b0000000000000000",
                                   sel3="0b101") == "0b1111111111111111"
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000",
                                   g16="0b1111111111111111", h16="0b0000000000000000",
                                   sel3="0b110") == "0b1111111111111111"
        assert _mux8way16.evaluate(a16="0b0000000000000000", b16="0b0000000000000000", c16="0b0000000000000000",
                                   d16="0b0000000000000000", e16="0b0000000000000000", f16="0b0000000000000000",
                                   g16="0b0000000000000000", h16="0b1111111111111111",
                                   sel3="0b111") == "0b1111111111111111"

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
        assert _alu.evaluate(x="0b0000000000000000", y="0b0000000000000000", zx="0b0", zy="0b0", nx="0b0", ny="0b0",
                             f="0b1", no="0b0") == ("0b0000000000000000", "0b1", "0b0")
        assert _alu.evaluate(x="0b0000000000000001", y="0b0000000000000001", zx="0b0", zy="0b0", nx="0b0", ny="0b0",
                             f="0b1", no="0b0") == ("0b0000000000000010", "0b0", "0b0")

        # ALU: zx/yx
        assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b1", zy="0b1", nx="0b0", ny="0b0",
                             f="0b1", no="0b0") == ("0b0000000000000000", "0b1", "0b0")
        assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b1", zy="0b0", nx="0b0", ny="0b0",
                             f="0b1", no="0b0") == ("0b1111111111111111", "0b0", "0b1")
        assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b0", zy="0b1", nx="0b0", ny="0b0",
                             f="0b1", no="0b0") == ("0b1111111111111111", "0b0", "0b1")

        # ALU: nx/ny
        assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b0", zy="0b0", nx="0b1", ny="0b1",
                             f="0b1", no="0b0") == ("0b0000000000000000", "0b1", "0b0")
        assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b0", zy="0b0", nx="0b1", ny="0b0",
                             f="0b1", no="0b0") == ("0b1111111111111111", "0b0", "0b1")
        assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b0", zy="0b0", nx="0b0", ny="0b1",
                             f="0b1", no="0b0") == ("0b1111111111111111", "0b0", "0b1")

        # ALU: and
        assert _alu.evaluate(x="0b0000000000000000", y="0b0000000000000000", zx="0b0", zy="0b0", nx="0b0", ny="0b0",
                             f="0b0", no="0b0") == ("0b0000000000000000", "0b1", "0b0")
        assert _alu.evaluate(x="0b0000000000000000", y="0b1111111111111111", zx="0b0", zy="0b0", nx="0b0", ny="0b0",
                             f="0b0", no="0b0") == ("0b0000000000000000", "0b1", "0b0")
        assert _alu.evaluate(x="0b1111111111111111", y="0b0000000000000000", zx="0b0", zy="0b0", nx="0b0", ny="0b0",
                             f="0b0", no="0b0") == ("0b0000000000000000", "0b1", "0b0")
        assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b0", zy="0b0", nx="0b0", ny="0b0",
                             f="0b0", no="0b0") == ("0b1111111111111111", "0b0", "0b1")

        # ALU: not(and)
        assert _alu.evaluate(x="0b1111111111111111", y="0b0000000000000000", zx="0b0", zy="0b0", nx="0b0", ny="0b0",
                             f="0b0", no="0b1") == ("0b1111111111111111", "0b0", "0b1")
        assert _alu.evaluate(x="0b1111111111111111", y="0b1111111111111111", zx="0b0", zy="0b0", nx="0b0", ny="0b0",
                             f="0b0", no="0b1") == ("0b0000000000000000", "0b1", "0b0")
        assert _alu.evaluate(x="0b0000000000000000", y="0b0000000000000000", zx="0b0", zy="0b0", nx="0b0", ny="0b0",
                             f="0b0", no="0b1") == ("0b1111111111111111", "0b0", "0b1")

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

        # RAM8: sequential set
        assert _ram8.evaluate(_in16="0b0000000000000000", load="0b1", addr3="0b000") == "0b0000000000000000"
        assert _ram8.evaluate(_in16="0b0000000000000001", load="0b1", addr3="0b001") == "0b0000000000000001"
        assert _ram8.evaluate(_in16="0b0000000000000010", load="0b1", addr3="0b010") == "0b0000000000000010"
        assert _ram8.evaluate(_in16="0b0000000000000011", load="0b1", addr3="0b011") == "0b0000000000000011"
        assert _ram8.evaluate(_in16="0b1000000000000000", load="0b1", addr3="0b100") == "0b1000000000000000"
        assert _ram8.evaluate(_in16="0b1010000000000000", load="0b1", addr3="0b101") == "0b1010000000000000"
        assert _ram8.evaluate(_in16="0b1100000000000000", load="0b1", addr3="0b110") == "0b1100000000000000"
        assert _ram8.evaluate(_in16="0b1110000000000000", load="0b1", addr3="0b111") == "0b1110000000000000"

        # RAM8: sequential load
        assert _ram8.evaluate(_in16="0b0000000000000000", load="0b0", addr3="0b000") == "0b0000000000000000"
        assert _ram8.evaluate(_in16="0b0000000000000001", load="0b0", addr3="0b001") == "0b0000000000000001"
        assert _ram8.evaluate(_in16="0b0000000000000010", load="0b0", addr3="0b010") == "0b0000000000000010"
        assert _ram8.evaluate(_in16="0b0000000000000011", load="0b0", addr3="0b011") == "0b0000000000000011"
        assert _ram8.evaluate(_in16="0b1000000000000000", load="0b0", addr3="0b100") == "0b1000000000000000"
        assert _ram8.evaluate(_in16="0b1010000000000000", load="0b0", addr3="0b101") == "0b1010000000000000"
        assert _ram8.evaluate(_in16="0b1100000000000000", load="0b0", addr3="0b110") == "0b1100000000000000"
        assert _ram8.evaluate(_in16="0b1110000000000000", load="0b0", addr3="0b111") == "0b1110000000000000"

        # RAM8: mixed load/set
        assert _ram8.evaluate(_in16="0b0000000000000000", load="0b1", addr3="0b000") == "0b0000000000000000"
        assert _ram8.evaluate(_in16="0b0000000000000001", load="0b1", addr3="0b001") == "0b0000000000000001"
        assert _ram8.evaluate(_in16="0b0000000000000010", load="0b1", addr3="0b010") == "0b0000000000000010"
        assert _ram8.evaluate(_in16="0b0000000000000011", load="0b1", addr3="0b011") == "0b0000000000000011"
        assert _ram8.evaluate(_in16="0b0000000000000000", load="0b0", addr3="0b000") == "0b0000000000000000"
        assert _ram8.evaluate(_in16="0b0000000000000001", load="0b0", addr3="0b001") == "0b0000000000000001"
        assert _ram8.evaluate(_in16="0b0000000000000010", load="0b0", addr3="0b010") == "0b0000000000000010"
        assert _ram8.evaluate(_in16="0b0000000000000011", load="0b0", addr3="0b011") == "0b0000000000000011"
        assert _ram8.evaluate(_in16="0b1000000000000000", load="0b1", addr3="0b100") == "0b1000000000000000"
        assert _ram8.evaluate(_in16="0b1010000000000000", load="0b1", addr3="0b101") == "0b1010000000000000"
        assert _ram8.evaluate(_in16="0b1100000000000000", load="0b1", addr3="0b110") == "0b1100000000000000"
        assert _ram8.evaluate(_in16="0b1110000000000000", load="0b1", addr3="0b111") == "0b1110000000000000"
        assert _ram8.evaluate(_in16="0b1000000000000000", load="0b0", addr3="0b100") == "0b1000000000000000"
        assert _ram8.evaluate(_in16="0b1010000000000000", load="0b0", addr3="0b101") == "0b1010000000000000"
        assert _ram8.evaluate(_in16="0b1100000000000000", load="0b0", addr3="0b110") == "0b1100000000000000"
        assert _ram8.evaluate(_in16="0b1110000000000000", load="0b0", addr3="0b111") == "0b1110000000000000"

        # RAM64: sequential set (000/XXX)
        assert _ram64.evaluate(_in16="0b0000000000000000", load="0b1", addr6="0b000000") == "0b0000000000000000"
        assert _ram64.evaluate(_in16="0b0000000000000001", load="0b1", addr6="0b000001") == "0b0000000000000001"
        assert _ram64.evaluate(_in16="0b0000000000000010", load="0b1", addr6="0b000010") == "0b0000000000000010"
        assert _ram64.evaluate(_in16="0b0000000000000011", load="0b1", addr6="0b000011") == "0b0000000000000011"
        assert _ram64.evaluate(_in16="0b1000000000000000", load="0b1", addr6="0b000100") == "0b1000000000000000"
        assert _ram64.evaluate(_in16="0b1010000000000000", load="0b1", addr6="0b000101") == "0b1010000000000000"
        assert _ram64.evaluate(_in16="0b1110000000000000", load="0b1", addr6="0b000110") == "0b1110000000000000"
        assert _ram64.evaluate(_in16="0b1111000000000000", load="0b1", addr6="0b000111") == "0b1111000000000000"

        # RAM64: sequential set (111/XXX)
        assert _ram64.evaluate(_in16="0b0000000000000000", load="0b1", addr6="0b111000") == "0b0000000000000000"
        assert _ram64.evaluate(_in16="0b0000000000000001", load="0b1", addr6="0b111001") == "0b0000000000000001"
        assert _ram64.evaluate(_in16="0b0000000000000010", load="0b1", addr6="0b111010") == "0b0000000000000010"
        assert _ram64.evaluate(_in16="0b0000000000000011", load="0b1", addr6="0b111011") == "0b0000000000000011"
        assert _ram64.evaluate(_in16="0b1000000000000000", load="0b1", addr6="0b111100") == "0b1000000000000000"
        assert _ram64.evaluate(_in16="0b1010000000000000", load="0b1", addr6="0b111101") == "0b1010000000000000"
        assert _ram64.evaluate(_in16="0b1110000000000000", load="0b1", addr6="0b111110") == "0b1110000000000000"
        assert _ram64.evaluate(_in16="0b1111000000000000", load="0b1", addr6="0b111111") == "0b1111000000000000"

        # RAM64: sequential load (000/XXX)
        assert _ram64.evaluate(_in16="0b0000000000000000", load="0b0", addr6="0b000000") == "0b0000000000000000"
        assert _ram64.evaluate(_in16="0b0000000000000001", load="0b0", addr6="0b000001") == "0b0000000000000001"
        assert _ram64.evaluate(_in16="0b0000000000000010", load="0b0", addr6="0b000010") == "0b0000000000000010"
        assert _ram64.evaluate(_in16="0b0000000000000011", load="0b0", addr6="0b000011") == "0b0000000000000011"
        assert _ram64.evaluate(_in16="0b1000000000000000", load="0b0", addr6="0b000100") == "0b1000000000000000"
        assert _ram64.evaluate(_in16="0b1010000000000000", load="0b0", addr6="0b000101") == "0b1010000000000000"
        assert _ram64.evaluate(_in16="0b1110000000000000", load="0b0", addr6="0b000110") == "0b1110000000000000"
        assert _ram64.evaluate(_in16="0b1111000000000000", load="0b0", addr6="0b000111") == "0b1111000000000000"

        # RAM64: sequential load (111/XXX)
        assert _ram64.evaluate(_in16="0b0000000000000000", load="0b0", addr6="0b111000") == "0b0000000000000000"
        assert _ram64.evaluate(_in16="0b0000000000000001", load="0b0", addr6="0b111001") == "0b0000000000000001"
        assert _ram64.evaluate(_in16="0b0000000000000010", load="0b0", addr6="0b111010") == "0b0000000000000010"
        assert _ram64.evaluate(_in16="0b0000000000000011", load="0b0", addr6="0b111011") == "0b0000000000000011"
        assert _ram64.evaluate(_in16="0b1000000000000000", load="0b0", addr6="0b111100") == "0b1000000000000000"
        assert _ram64.evaluate(_in16="0b1010000000000000", load="0b0", addr6="0b111101") == "0b1010000000000000"
        assert _ram64.evaluate(_in16="0b1110000000000000", load="0b0", addr6="0b111110") == "0b1110000000000000"
        assert _ram64.evaluate(_in16="0b1111000000000000", load="0b0", addr6="0b111111") == "0b1111000000000000"

        # RAM512: sequential set (000/XXX)
        assert _ram512.evaluate(_in16="0b0000000000000000", load="0b1", addr9="0b000000000") == "0b0000000000000000"
        assert _ram512.evaluate(_in16="0b0000000000000001", load="0b1", addr9="0b000000001") == "0b0000000000000001"
        assert _ram512.evaluate(_in16="0b0000000000000010", load="0b1", addr9="0b000000010") == "0b0000000000000010"
        assert _ram512.evaluate(_in16="0b0000000000000011", load="0b1", addr9="0b000000111") == "0b0000000000000011"
        assert _ram512.evaluate(_in16="0b1000000000000000", load="0b1", addr9="0b000001100") == "0b1000000000000000"
        assert _ram512.evaluate(_in16="0b1010000000000000", load="0b1", addr9="0b000011101") == "0b1010000000000000"
        assert _ram512.evaluate(_in16="0b1110000000000000", load="0b1", addr9="0b000111110") == "0b1110000000000000"
        assert _ram512.evaluate(_in16="0b1111000000000000", load="0b1", addr9="0b001111111") == "0b1111000000000000"

        # RAM512: sequential set (111/XXX)
        assert _ram512.evaluate(_in16="0b0000000000000000", load="0b1", addr9="0b111111000") == "0b0000000000000000"
        assert _ram512.evaluate(_in16="0b0000000000000001", load="0b1", addr9="0b111110001") == "0b0000000000000001"
        assert _ram512.evaluate(_in16="0b0000000000000010", load="0b1", addr9="0b111100010") == "0b0000000000000010"
        assert _ram512.evaluate(_in16="0b0000000000000011", load="0b1", addr9="0b111000011") == "0b0000000000000011"
        assert _ram512.evaluate(_in16="0b1000000000000000", load="0b1", addr9="0b110000000") == "0b1000000000000000"
        assert _ram512.evaluate(_in16="0b1010000000000000", load="0b1", addr9="0b111111101") == "0b1010000000000000"
        assert _ram512.evaluate(_in16="0b1110000000000000", load="0b1", addr9="0b111111110") == "0b1110000000000000"
        assert _ram512.evaluate(_in16="0b1111000000000000", load="0b1", addr9="0b111111111") == "0b1111000000000000"

        # RAM512: sequential load (000/XXX)
        assert _ram512.evaluate(_in16="0b0000000000000000", load="0b0", addr9="0b000000000") == "0b0000000000000000"
        assert _ram512.evaluate(_in16="0b0000000000000001", load="0b0", addr9="0b000000001") == "0b0000000000000001"
        assert _ram512.evaluate(_in16="0b0000000000000010", load="0b0", addr9="0b000000010") == "0b0000000000000010"
        assert _ram512.evaluate(_in16="0b0000000000000011", load="0b0", addr9="0b000000111") == "0b0000000000000011"
        assert _ram512.evaluate(_in16="0b1000000000000000", load="0b0", addr9="0b000001100") == "0b1000000000000000"
        assert _ram512.evaluate(_in16="0b1010000000000000", load="0b0", addr9="0b000011101") == "0b1010000000000000"
        assert _ram512.evaluate(_in16="0b1110000000000000", load="0b0", addr9="0b000111110") == "0b1110000000000000"
        assert _ram512.evaluate(_in16="0b1111000000000000", load="0b0", addr9="0b001111111") == "0b1111000000000000"

        # RAM512: sequential load (111/XXX)
        assert _ram512.evaluate(_in16="0b0000000000000000", load="0b0", addr9="0b111111000") == "0b0000000000000000"
        assert _ram512.evaluate(_in16="0b0000000000000001", load="0b0", addr9="0b111110001") == "0b0000000000000001"
        assert _ram512.evaluate(_in16="0b0000000000000010", load="0b0", addr9="0b111100010") == "0b0000000000000010"
        assert _ram512.evaluate(_in16="0b0000000000000011", load="0b0", addr9="0b111000011") == "0b0000000000000011"
        assert _ram512.evaluate(_in16="0b1000000000000000", load="0b0", addr9="0b110000000") == "0b1000000000000000"
        assert _ram512.evaluate(_in16="0b1010000000000000", load="0b0", addr9="0b111111101") == "0b1010000000000000"
        assert _ram512.evaluate(_in16="0b1110000000000000", load="0b0", addr9="0b111111110") == "0b1110000000000000"
        assert _ram512.evaluate(_in16="0b1111000000000000", load="0b0", addr9="0b111111111") == "0b1111000000000000"

        # RAM4K: sequential set (000/XXX)
        assert _ram4k.evaluate(_in16="0b0000000000000000", load="0b1", addr12="0b000000000000") == "0b0000000000000000"
        assert _ram4k.evaluate(_in16="0b0000000000000001", load="0b1", addr12="0b000000000001") == "0b0000000000000001"
        assert _ram4k.evaluate(_in16="0b0000000000000010", load="0b1", addr12="0b000000000010") == "0b0000000000000010"
        assert _ram4k.evaluate(_in16="0b0000000000000011", load="0b1", addr12="0b000000000111") == "0b0000000000000011"
        assert _ram4k.evaluate(_in16="0b1000000000000000", load="0b1", addr12="0b000000001100") == "0b1000000000000000"
        assert _ram4k.evaluate(_in16="0b1010000000000000", load="0b1", addr12="0b000000011101") == "0b1010000000000000"
        assert _ram4k.evaluate(_in16="0b1110000000000000", load="0b1", addr12="0b000001111110") == "0b1110000000000000"
        assert _ram4k.evaluate(_in16="0b1111000000000000", load="0b1", addr12="0b000011111111") == "0b1111000000000000"

        # RAM4K: sequential set (111/XXX)
        assert _ram4k.evaluate(_in16="0b0000000000000000", load="0b1", addr12="0b111111111000") == "0b0000000000000000"
        assert _ram4k.evaluate(_in16="0b0000000000000001", load="0b1", addr12="0b111111110001") == "0b0000000000000001"
        assert _ram4k.evaluate(_in16="0b0000000000000010", load="0b1", addr12="0b111111100010") == "0b0000000000000010"
        assert _ram4k.evaluate(_in16="0b0000000000000011", load="0b1", addr12="0b111111000011") == "0b0000000000000011"
        assert _ram4k.evaluate(_in16="0b1000000000000000", load="0b1", addr12="0b111110000000") == "0b1000000000000000"
        assert _ram4k.evaluate(_in16="0b1010000000000000", load="0b1", addr12="0b111111111101") == "0b1010000000000000"
        assert _ram4k.evaluate(_in16="0b1110000000000000", load="0b1", addr12="0b111111111110") == "0b1110000000000000"
        assert _ram4k.evaluate(_in16="0b1111000000000000", load="0b1", addr12="0b111111111111") == "0b1111000000000000"

        # RAM4K: sequential load (000/XXX)
        assert _ram4k.evaluate(_in16="0b0000000000000000", load="0b0", addr12="0b000000000000") == "0b0000000000000000"
        assert _ram4k.evaluate(_in16="0b0000000000000001", load="0b0", addr12="0b000000000001") == "0b0000000000000001"
        assert _ram4k.evaluate(_in16="0b0000000000000010", load="0b0", addr12="0b000000000010") == "0b0000000000000010"
        assert _ram4k.evaluate(_in16="0b0000000000000011", load="0b0", addr12="0b000000000111") == "0b0000000000000011"
        assert _ram4k.evaluate(_in16="0b1000000000000000", load="0b0", addr12="0b000000001100") == "0b1000000000000000"
        assert _ram4k.evaluate(_in16="0b1010000000000000", load="0b0", addr12="0b000000011101") == "0b1010000000000000"
        assert _ram4k.evaluate(_in16="0b1110000000000000", load="0b0", addr12="0b000001111110") == "0b1110000000000000"
        assert _ram4k.evaluate(_in16="0b1111000000000000", load="0b0", addr12="0b000011111111") == "0b1111000000000000"

        # RAM4K: sequential load (111/XXX)
        assert _ram4k.evaluate(_in16="0b0000000000000000", load="0b0", addr12="0b111111111000") == "0b0000000000000000"
        assert _ram4k.evaluate(_in16="0b0000000000000001", load="0b0", addr12="0b111111110001") == "0b0000000000000001"
        assert _ram4k.evaluate(_in16="0b0000000000000010", load="0b0", addr12="0b111111100010") == "0b0000000000000010"
        assert _ram4k.evaluate(_in16="0b0000000000000011", load="0b0", addr12="0b111111000011") == "0b0000000000000011"
        assert _ram4k.evaluate(_in16="0b1000000000000000", load="0b0", addr12="0b111110000000") == "0b1000000000000000"
        assert _ram4k.evaluate(_in16="0b1010000000000000", load="0b0", addr12="0b111111111101") == "0b1010000000000000"
        assert _ram4k.evaluate(_in16="0b1110000000000000", load="0b0", addr12="0b111111111110") == "0b1110000000000000"
        assert _ram4k.evaluate(_in16="0b1111000000000000", load="0b0", addr12="0b111111111111") == "0b1111000000000000"

        # RAM16K: sequential set (000/XXX)
        assert _ram16k.evaluate(_in16="0b0000000000000000", load="0b1",
                                addr14="0b00000000000000") == "0b0000000000000000"
        assert _ram16k.evaluate(_in16="0b0000000000000001", load="0b1",
                                addr14="0b00000000000001") == "0b0000000000000001"
        assert _ram16k.evaluate(_in16="0b0000000000000010", load="0b1",
                                addr14="0b00000000000010") == "0b0000000000000010"
        assert _ram16k.evaluate(_in16="0b0000000000000011", load="0b1",
                                addr14="0b00000000000111") == "0b0000000000000011"
        assert _ram16k.evaluate(_in16="0b1000000000000000", load="0b1",
                                addr14="0b00000000001100") == "0b1000000000000000"
        assert _ram16k.evaluate(_in16="0b1010000000000000", load="0b1",
                                addr14="0b00000000011101") == "0b1010000000000000"
        assert _ram16k.evaluate(_in16="0b1110000000000000", load="0b1",
                                addr14="0b00000001111110") == "0b1110000000000000"
        assert _ram16k.evaluate(_in16="0b1111000000000000", load="0b1",
                                addr14="0b00000011111111") == "0b1111000000000000"

        # RAM16K: sequential set (111/XXX)
        assert _ram16k.evaluate(_in16="0b0000000000000000", load="0b1",
                                addr14="0b11111111111000") == "0b0000000000000000"
        assert _ram16k.evaluate(_in16="0b0000000000000001", load="0b1",
                                addr14="0b11111111110001") == "0b0000000000000001"
        assert _ram16k.evaluate(_in16="0b0000000000000010", load="0b1",
                                addr14="0b11111111100010") == "0b0000000000000010"
        assert _ram16k.evaluate(_in16="0b0000000000000011", load="0b1",
                                addr14="0b11111111000011") == "0b0000000000000011"
        assert _ram16k.evaluate(_in16="0b1000000000000000", load="0b1",
                                addr14="0b11111110000000") == "0b1000000000000000"
        assert _ram16k.evaluate(_in16="0b1010000000000000", load="0b1",
                                addr14="0b11111111111101") == "0b1010000000000000"
        assert _ram16k.evaluate(_in16="0b1110000000000000", load="0b1",
                                addr14="0b11111111111110") == "0b1110000000000000"
        assert _ram16k.evaluate(_in16="0b1111000000000000", load="0b1",
                                addr14="0b11111111111111") == "0b1111000000000000"

        # RAM16K: sequential load (000/XXX)
        assert _ram16k.evaluate(_in16="0b0000000000000000", load="0b0",
                                addr14="0b00000000000000") == "0b0000000000000000"
        assert _ram16k.evaluate(_in16="0b0000000000000001", load="0b0",
                                addr14="0b00000000000001") == "0b0000000000000001"
        assert _ram16k.evaluate(_in16="0b0000000000000010", load="0b0",
                                addr14="0b00000000000010") == "0b0000000000000010"
        assert _ram16k.evaluate(_in16="0b0000000000000011", load="0b0",
                                addr14="0b00000000000111") == "0b0000000000000011"
        assert _ram16k.evaluate(_in16="0b1000000000000000", load="0b0",
                                addr14="0b00000000001100") == "0b1000000000000000"
        assert _ram16k.evaluate(_in16="0b1010000000000000", load="0b0",
                                addr14="0b00000000011101") == "0b1010000000000000"
        assert _ram16k.evaluate(_in16="0b1110000000000000", load="0b0",
                                addr14="0b00000001111110") == "0b1110000000000000"
        assert _ram16k.evaluate(_in16="0b1111000000000000", load="0b0",
                                addr14="0b00000011111111") == "0b1111000000000000"

        # RAM16K: sequential load (111/XXX)
        assert _ram16k.evaluate(_in16="0b0000000000000000", load="0b0",
                                addr14="0b11111111111000") == "0b0000000000000000"
        assert _ram16k.evaluate(_in16="0b0000000000000001", load="0b0",
                                addr14="0b11111111110001") == "0b0000000000000001"
        assert _ram16k.evaluate(_in16="0b0000000000000010", load="0b0",
                                addr14="0b11111111100010") == "0b0000000000000010"
        assert _ram16k.evaluate(_in16="0b0000000000000011", load="0b0",
                                addr14="0b11111111000011") == "0b0000000000000011"
        assert _ram16k.evaluate(_in16="0b1000000000000000", load="0b0",
                                addr14="0b11111110000000") == "0b1000000000000000"
        assert _ram16k.evaluate(_in16="0b1010000000000000", load="0b0",
                                addr14="0b11111111111101") == "0b1010000000000000"
        assert _ram16k.evaluate(_in16="0b1110000000000000", load="0b0",
                                addr14="0b11111111111110") == "0b1110000000000000"
        assert _ram16k.evaluate(_in16="0b1111000000000000", load="0b0",
                                addr14="0b11111111111111") == "0b1111000000000000"

        # 3 bit input, 1 bit select, 2 x 3 bit output dmux
        assert _dmux3.evaluate(_in3="0b000", sel="0b0") == ("0b000", "0b000")
        assert _dmux3.evaluate(_in3="0b001", sel="0b0") == ("0b001", "0b000")
        assert _dmux3.evaluate(_in3="0b010", sel="0b0") == ("0b010", "0b000")
        assert _dmux3.evaluate(_in3="0b011", sel="0b0") == ("0b011", "0b000")
        assert _dmux3.evaluate(_in3="0b100", sel="0b0") == ("0b100", "0b000")
        assert _dmux3.evaluate(_in3="0b101", sel="0b0") == ("0b101", "0b000")
        assert _dmux3.evaluate(_in3="0b110", sel="0b0") == ("0b110", "0b000")
        assert _dmux3.evaluate(_in3="0b111", sel="0b0") == ("0b111", "0b000")
        assert _dmux3.evaluate(_in3="0b000", sel="0b1") == ("0b000", "0b000")
        assert _dmux3.evaluate(_in3="0b001", sel="0b1") == ("0b000", "0b001")
        assert _dmux3.evaluate(_in3="0b010", sel="0b1") == ("0b000", "0b010")
        assert _dmux3.evaluate(_in3="0b011", sel="0b1") == ("0b000", "0b011")
        assert _dmux3.evaluate(_in3="0b100", sel="0b1") == ("0b000", "0b100")
        assert _dmux3.evaluate(_in3="0b101", sel="0b1") == ("0b000", "0b101")
        assert _dmux3.evaluate(_in3="0b110", sel="0b1") == ("0b000", "0b110")
        assert _dmux3.evaluate(_in3="0b111", sel="0b1") == ("0b000", "0b111")

        # Dmux4Way3
        assert _dmux4way3.evaluate(_in3="0b000", sel2="0b00") == ("0b000", "0b000", "0b000", "0b000")
        assert _dmux4way3.evaluate(_in3="0b000", sel2="0b00") == ("0b000", "0b000", "0b000", "0b000")
        assert _dmux4way3.evaluate(_in3="0b000", sel2="0b00") == ("0b000", "0b000", "0b000", "0b000")
        assert _dmux4way3.evaluate(_in3="0b000", sel2="0b00") == ("0b000", "0b000", "0b000", "0b000")
        assert _dmux4way3.evaluate(_in3="0b111", sel2="0b00") == ("0b111", "0b000", "0b000", "0b000")
        assert _dmux4way3.evaluate(_in3="0b111", sel2="0b01") == ("0b000", "0b111", "0b000", "0b000")
        assert _dmux4way3.evaluate(_in3="0b111", sel2="0b10") == ("0b000", "0b000", "0b111", "0b000")
        assert _dmux4way3.evaluate(_in3="0b111", sel2="0b11") == ("0b000", "0b000", "0b000", "0b111")

        # SCREEN: sequential set (000/XXX)
        assert _screen.evaluate(_in16="0b0000000000000000", load="0b1",
                                addr13="0b0000000000000") == "0b0000000000000000"
        assert _screen.evaluate(_in16="0b0000000000000001", load="0b1",
                                addr13="0b0000000000001") == "0b0000000000000001"
        assert _screen.evaluate(_in16="0b0000000000000010", load="0b1",
                                addr13="0b0000000000010") == "0b0000000000000010"
        assert _screen.evaluate(_in16="0b0000000000000011", load="0b1",
                                addr13="0b0000000000111") == "0b0000000000000011"
        assert _screen.evaluate(_in16="0b1000000000000000", load="0b1",
                                addr13="0b0000000001100") == "0b1000000000000000"
        assert _screen.evaluate(_in16="0b1010000000000000", load="0b1",
                                addr13="0b0000000011101") == "0b1010000000000000"
        assert _screen.evaluate(_in16="0b1110000000000000", load="0b1",
                                addr13="0b0000001111110") == "0b1110000000000000"
        assert _screen.evaluate(_in16="0b1111000000000000", load="0b1",
                                addr13="0b0000011111111") == "0b1111000000000000"

        # SCREEN: sequential set (111/XXX)
        assert _screen.evaluate(_in16="0b0000000000000000", load="0b1",
                                addr13="0b1111111111000") == "0b0000000000000000"
        assert _screen.evaluate(_in16="0b0000000000000001", load="0b1",
                                addr13="0b1111111110001") == "0b0000000000000001"
        assert _screen.evaluate(_in16="0b0000000000000010", load="0b1",
                                addr13="0b1111111100010") == "0b0000000000000010"
        assert _screen.evaluate(_in16="0b0000000000000011", load="0b1",
                                addr13="0b1111111000011") == "0b0000000000000011"
        assert _screen.evaluate(_in16="0b1000000000000000", load="0b1",
                                addr13="0b1111110000000") == "0b1000000000000000"
        assert _screen.evaluate(_in16="0b1010000000000000", load="0b1",
                                addr13="0b1111111111101") == "0b1010000000000000"
        assert _screen.evaluate(_in16="0b1110000000000000", load="0b1",
                                addr13="0b1111111111110") == "0b1110000000000000"
        assert _screen.evaluate(_in16="0b1111000000000000", load="0b1",
                                addr13="0b1111111111111") == "0b1111000000000000"

        # SCREEN: sequential load (000/XXX)
        assert _screen.evaluate(_in16="0b0000000000000000", load="0b0",
                                addr13="0b0000000000000") == "0b0000000000000000"
        assert _screen.evaluate(_in16="0b0000000000000001", load="0b0",
                                addr13="0b0000000000001") == "0b0000000000000001"
        assert _screen.evaluate(_in16="0b0000000000000010", load="0b0",
                                addr13="0b0000000000010") == "0b0000000000000010"
        assert _screen.evaluate(_in16="0b0000000000000011", load="0b0",
                                addr13="0b0000000000111") == "0b0000000000000011"
        assert _screen.evaluate(_in16="0b1000000000000000", load="0b0",
                                addr13="0b0000000001100") == "0b1000000000000000"
        assert _screen.evaluate(_in16="0b1010000000000000", load="0b0",
                                addr13="0b0000000011101") == "0b1010000000000000"
        assert _screen.evaluate(_in16="0b1110000000000000", load="0b0",
                                addr13="0b0000001111110") == "0b1110000000000000"
        assert _screen.evaluate(_in16="0b1111000000000000", load="0b0",
                                addr13="0b0000011111111") == "0b1111000000000000"

        # SCREEN: sequential load (111/XXX)
        assert _screen.evaluate(_in16="0b0000000000000000", load="0b0",
                                addr13="0b1111111111000") == "0b0000000000000000"
        assert _screen.evaluate(_in16="0b0000000000000001", load="0b0",
                                addr13="0b1111111110001") == "0b0000000000000001"
        assert _screen.evaluate(_in16="0b0000000000000010", load="0b0",
                                addr13="0b1111111100010") == "0b0000000000000010"
        assert _screen.evaluate(_in16="0b0000000000000011", load="0b0",
                                addr13="0b1111111000011") == "0b0000000000000011"
        assert _screen.evaluate(_in16="0b1000000000000000", load="0b0",
                                addr13="0b1111110000000") == "0b1000000000000000"
        assert _screen.evaluate(_in16="0b1010000000000000", load="0b0",
                                addr13="0b1111111111101") == "0b1010000000000000"
        assert _screen.evaluate(_in16="0b1110000000000000", load="0b0",
                                addr13="0b1111111111110") == "0b1110000000000000"
        assert _screen.evaluate(_in16="0b1111000000000000", load="0b0",
                                addr13="0b1111111111111") == "0b1111000000000000"

        # 16K+8K+1 memory block for RAM, Screen, Keyboard address ranges respectively
        assert _memory.evaluate(_in16="0b1111000000000000", load="0b1",
                                addr15="0b001111111110000") == "0b1111000000000000"  # RAM (RAM16K)
        assert _memory.evaluate(_in16="0b0000111100000000", load="0b1",
                                addr15="0b011111100001111") == "0b0000111100000000"  # RAM (RAM16K)
        assert _memory.evaluate(_in16="0b1111000011110000", load="0b1",
                                addr15="0b101000011111111") == "0b1111000011110000"  # SCREEN (RAM8K)
        assert _memory.evaluate(_in16="0b1111000000001111", load="0b1",
                                addr15="0b110000111111111") == "0b1111000000001111"  # KEYBOARD (Register)
        assert _memory.evaluate(_in16="0b1111000000000000", load="0b0",
                                addr15="0b001111111110000") == "0b1111000000000000"  # RAM (RAM16K)
        assert _memory.evaluate(_in16="0b1111000000000000", load="0b0",
                                addr15="0b011111100001111") == "0b0000111100000000"  # RAM (RAM16K)
        assert _memory.evaluate(_in16="0b1111000000000000", load="0b0",
                                addr15="0b101000011111111") == "0b1111000011110000"  # SCREEN (RAM8K)
        assert _memory.evaluate(_in16="0b1111000000000000", load="0b0",
                                addr15="0b110000111111111") == "0b1111000000001111"  # KEYBOARD (Register)

        # CPU: instruction, m_in, reset = m_out, writeM, a_out, pc_out, d_out
        # a instructions (m_out is random ALU bits on A inst)
        assert _cpu.evaluate(_in16="0b0000000000000000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000000000", "0b0000000000000001",
             "0b0000000000000000")  # @0000000000000000
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000000010",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b0000000011110000", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111100001111", "0b0", "0b0000000011110000", "0b0000000000000011",
             "0b0000000000000000")  # @0000000011110000
        assert _cpu.evaluate(_in16="0b0000111100000000", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0000111100000000", "0b0000000000000100",
             "0b0000000000000000")  # @0000111100000000
        assert _cpu.evaluate(_in16="0b0001111111111111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0001111111111111", "0b0000000000000101",
             "0b0000000000000000")  # @0001111111111111

        # c instructions: comp bits (write dest, no jump)
        assert _cpu.evaluate(_in16="0b111" + "0" + "101010" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0001111111111111", "0b0000000000000110", "0b0000000000000000")  # D=0
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0001111111111111", "0b0000000000000111", "0b0000000000000001")  # D=1
        assert _cpu.evaluate(_in16="0b111" + "0" + "111010" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0001111111111111", "0b0000000000001000", "0b1111111111111111")  # D=-1
        assert _cpu.evaluate(_in16="0b111" + "0" + "001100" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0001111111111111", "0b0000000000001001", "0b1111111111111111")  # D=D
        assert _cpu.evaluate(_in16="0b111" + "0" + "110000" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0001111111111111", "0b0", "0b0001111111111111", "0b0000000000001010", "0b0001111111111111")  # D=A
        assert _cpu.evaluate(_in16="0b111" + "1" + "110000" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0001111111111111", "0b0000000000001011", "0b0000000000000000")  # D=M
        assert _cpu.evaluate(_in16="0b111" + "0" + "001101" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0001111111111111", "0b0000000000001100", "0b1111111111111111")  # D=!D
        assert _cpu.evaluate(_in16="0b111" + "0" + "110001" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b1110000000000000", "0b0", "0b0001111111111111", "0b0000000000001101", "0b1110000000000000")  # D=!A
        assert _cpu.evaluate(_in16="0b111" + "1" + "110001" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0001111111111111", "0b0000000000001110", "0b1111111111111111")  # D=!M
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0001111111111111", "0b0000000000001111", "0b0000000000000001")  # D=1
        assert _cpu.evaluate(_in16="0b111" + "1" + "001111" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0001111111111111", "0b0000000000010000", "0b1111111111111111")  # D=-D
        assert _cpu.evaluate(_in16="0b111" + "0" + "110011" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b1110000000000001", "0b0", "0b0001111111111111", "0b0000000000010001", "0b1110000000000001")  # D=-A
        assert _cpu.evaluate(_in16="0b111" + "1" + "110011" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0001111111111111", "0b0000000000010010", "0b0000000000000000")  # D=-M
        assert _cpu.evaluate(_in16="0b111" + "0" + "011111" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0001111111111111", "0b0000000000010011", "0b0000000000000001")  # D=D+1
        assert _cpu.evaluate(_in16="0b111" + "0" + "110111" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0010000000000000", "0b0", "0b0001111111111111", "0b0000000000010100", "0b0010000000000000")  # D=A+1
        assert _cpu.evaluate(_in16="0b111" + "1" + "110111" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0001111111111111", "0b0000000000010101", "0b0000000000000001")  # D=M+1
        assert _cpu.evaluate(_in16="0b111" + "0" + "001110" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0001111111111111", "0b0000000000010110", "0b0000000000000000")  # D=D-1
        assert _cpu.evaluate(_in16="0b111" + "0" + "110010" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0001111111111110", "0b0", "0b0001111111111111", "0b0000000000010111", "0b0001111111111110")  # D=A-1
        assert _cpu.evaluate(_in16="0b111" + "1" + "110010" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0001111111111111", "0b0000000000011000", "0b1111111111111111")  # D=M-1
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0001111111111111", "0b0000000000011001", "0b0000000000000001")  # D=1
        assert _cpu.evaluate(_in16="0b111" + "0" + "000010" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0010000000000000", "0b0", "0b0001111111111111", "0b0000000000011010", "0b0010000000000000")  # D=D+A
        assert _cpu.evaluate(_in16="0b111" + "1" + "000010" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0010000000000000", "0b0", "0b0001111111111111", "0b0000000000011011", "0b0010000000000000")  # D=D+M
        assert _cpu.evaluate(_in16="0b111" + "0" + "010011" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0001111111111111", "0b0000000000011100", "0b0000000000000001")  # D=D-A
        assert _cpu.evaluate(_in16="0b111" + "1" + "010011" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0001111111111111", "0b0000000000011101", "0b0000000000000001")  # D=D-M
        assert _cpu.evaluate(_in16="0b111" + "0" + "000111" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0001111111111110", "0b0", "0b0001111111111111", "0b0000000000011110", "0b0001111111111110")  # D=A-D
        assert _cpu.evaluate(_in16="0b111" + "1" + "000111" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b1110000000000010", "0b0", "0b0001111111111111", "0b0000000000011111", "0b1110000000000010")  # D=M-D
        assert _cpu.evaluate(_in16="0b111" + "0" + "000000" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000010", "0b0", "0b0001111111111111", "0b0000000000100000", "0b0000000000000010")  # D=D&A
        assert _cpu.evaluate(_in16="0b111" + "1" + "000000" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0001111111111111", "0b0000000000100001", "0b0000000000000000")  # D=D&M
        assert _cpu.evaluate(_in16="0b111" + "0" + "010101" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0001111111111111", "0b0", "0b0001111111111111", "0b0000000000100010", "0b0001111111111111")  # D=D|A
        assert _cpu.evaluate(_in16="0b111" + "1" + "010101" + "010" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0001111111111111", "0b0", "0b0001111111111111", "0b0000000000100011", "0b0001111111111111")  # D=D|M

        assert _cpu.evaluate(_in16="0b111" + "0" + "101010" + "001" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b1", "0b0001111111111111", "0b0000000000100100", "0b0001111111111111")  # M=0
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "001" + "000", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b1", "0b0001111111111111", "0b0000000000100101", "0b0001111111111111")  # M=1
        assert _cpu.evaluate(_in16="0b111" + "0" + "111010" + "001" + "000", b16="0b0000000000000001", reset="0b0") == \
            ("0b1111111111111111", "0b1", "0b0001111111111111", "0b0000000000100110", "0b0001111111111111")  # M=-1
        assert _cpu.evaluate(_in16="0b111" + "0" + "001100" + "001" + "000", b16="0b1111111111111111", reset="0b0") == \
            ("0b0001111111111111", "0b1", "0b0001111111111111", "0b0000000000100111", "0b0001111111111111")  # M=D
        assert _cpu.evaluate(_in16="0b111" + "0" + "110000" + "001" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0001111111111111", "0b1", "0b0001111111111111", "0b0000000000101000", "0b0001111111111111")  # M=A
        assert _cpu.evaluate(_in16="0b111" + "1" + "110000" + "001" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0001111111111111", "0b1", "0b0001111111111111", "0b0000000000101001", "0b0001111111111111")  # M=M
        assert _cpu.evaluate(_in16="0b111" + "0" + "001101" + "001" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b1110000000000000", "0b1", "0b0001111111111111", "0b0000000000101010", "0b0001111111111111")  # M=!D
        assert _cpu.evaluate(_in16="0b111" + "0" + "110001" + "001" + "000", b16="0b1110000000000000", reset="0b0") == \
            ("0b1110000000000000", "0b1", "0b0001111111111111", "0b0000000000101011", "0b0001111111111111")  # M=!A
        assert _cpu.evaluate(_in16="0b111" + "1" + "110001" + "001" + "000", b16="0b1110000000000000", reset="0b0") == \
            ("0b0001111111111111", "0b1", "0b0001111111111111", "0b0000000000101100", "0b0001111111111111")  # M=!M
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "001" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0000000000000001", "0b1", "0b0001111111111111", "0b0000000000101101", "0b0001111111111111")  # M=1
        assert _cpu.evaluate(_in16="0b111" + "1" + "001111" + "001" + "000", b16="0b0000000000000001", reset="0b0") == \
            ("0b1110000000000001", "0b1", "0b0001111111111111", "0b0000000000101110", "0b0001111111111111")  # M=-D
        assert _cpu.evaluate(_in16="0b111" + "0" + "110011" + "001" + "000", b16="0b1110000000000001", reset="0b0") == \
            ("0b1110000000000001", "0b1", "0b0001111111111111", "0b0000000000101111", "0b0001111111111111")  # M=-A
        assert _cpu.evaluate(_in16="0b111" + "1" + "110011" + "001" + "000", b16="0b1110000000000001", reset="0b0") == \
            ("0b0001111111111111", "0b1", "0b0001111111111111", "0b0000000000110000", "0b0001111111111111")  # M=-M
        assert _cpu.evaluate(_in16="0b111" + "0" + "011111" + "001" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0010000000000000", "0b1", "0b0001111111111111", "0b0000000000110001", "0b0001111111111111")  # M=D+1
        assert _cpu.evaluate(_in16="0b111" + "0" + "110111" + "001" + "000", b16="0b0010000000000000", reset="0b0") == \
            ("0b0010000000000000", "0b1", "0b0001111111111111", "0b0000000000110010", "0b0001111111111111")  # M=A+1
        assert _cpu.evaluate(_in16="0b111" + "1" + "110111" + "001" + "000", b16="0b0010000000000000", reset="0b0") == \
            ("0b0010000000000001", "0b1", "0b0001111111111111", "0b0000000000110011", "0b0001111111111111")  # M=M+1
        assert _cpu.evaluate(_in16="0b111" + "0" + "001110" + "001" + "000", b16="0b0010000000000001", reset="0b0") == \
            ("0b0001111111111110", "0b1", "0b0001111111111111", "0b0000000000110100", "0b0001111111111111")  # M=D-1
        assert _cpu.evaluate(_in16="0b111" + "0" + "110010" + "001" + "000", b16="0b0001111111111110", reset="0b0") == \
            ("0b0001111111111110", "0b1", "0b0001111111111111", "0b0000000000110101", "0b0001111111111111")  # M=A-1
        assert _cpu.evaluate(_in16="0b111" + "1" + "110010" + "001" + "000", b16="0b0001111111111110", reset="0b0") == \
            ("0b0001111111111101", "0b1", "0b0001111111111111", "0b0000000000110110", "0b0001111111111111")  # M=M-1
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "010" + "000", b16="0b0001111111111101", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0001111111111111", "0b0000000000110111", "0b0000000000000001")  # D=1
        assert _cpu.evaluate(_in16="0b111" + "0" + "000010" + "001" + "000", b16="0b0000000000000001", reset="0b0") == \
            ("0b0010000000000000", "0b1", "0b0001111111111111", "0b0000000000111000", "0b0000000000000001")  # M=D+A
        assert _cpu.evaluate(_in16="0b111" + "1" + "000010" + "001" + "000", b16="0b0010000000000000", reset="0b0") == \
            ("0b0010000000000001", "0b1", "0b0001111111111111", "0b0000000000111001", "0b0000000000000001")  # M=D+M
        assert _cpu.evaluate(_in16="0b111" + "0" + "010011" + "001" + "000", b16="0b0010000000000001", reset="0b0") == \
            ("0b1110000000000010", "0b1", "0b0001111111111111", "0b0000000000111010", "0b0000000000000001")  # M=D-A
        assert _cpu.evaluate(_in16="0b111" + "1" + "010011" + "001" + "000", b16="0b1110000000000010", reset="0b0") == \
            ("0b0001111111111111", "0b1", "0b0001111111111111", "0b0000000000111011", "0b0000000000000001")  # M=D-M
        assert _cpu.evaluate(_in16="0b111" + "0" + "000111" + "001" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0001111111111110", "0b1", "0b0001111111111111", "0b0000000000111100", "0b0000000000000001")  # M=A-D
        assert _cpu.evaluate(_in16="0b111" + "1" + "000111" + "001" + "000", b16="0b0001111111111110", reset="0b0") == \
            ("0b0001111111111101", "0b1", "0b0001111111111111", "0b0000000000111101", "0b0000000000000001")  # M=M-D
        assert _cpu.evaluate(_in16="0b111" + "0" + "000000" + "001" + "000", b16="0b0001111111111101", reset="0b0") == \
            ("0b0000000000000001", "0b1", "0b0001111111111111", "0b0000000000111110", "0b0000000000000001")  # M=D&A
        assert _cpu.evaluate(_in16="0b111" + "1" + "000000" + "001" + "000", b16="0b0000000000000001", reset="0b0") == \
            ("0b0000000000000001", "0b1", "0b0001111111111111", "0b0000000000111111", "0b0000000000000001")  # M=D&M
        assert _cpu.evaluate(_in16="0b111" + "0" + "010101" + "001" + "000", b16="0b0000000000000001", reset="0b0") == \
            ("0b0001111111111111", "0b1", "0b0001111111111111", "0b0000000001000000", "0b0000000000000001")  # M=D|A
        assert _cpu.evaluate(_in16="0b111" + "1" + "010101" + "001" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0001111111111111", "0b1", "0b0001111111111111", "0b0000000001000001", "0b0000000000000001")  # M=D|M

        assert _cpu.evaluate(_in16="0b111" + "0" + "101010" + "100" + "000", b16="0b0000000000000111", reset="0b0") == \
            ("0b0000000000000111", "0b0", "0b0000000000000000", "0b0000000001000010", "0b0000000000000001")  # A=0
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "100" + "000", b16="0b0000011100000000", reset="0b0") == \
            ("0b0000011100000000", "0b0", "0b0000000000000001", "0b0000000001000011", "0b0000000000000001")  # A=1
        assert _cpu.evaluate(_in16="0b111" + "0" + "111010" + "100" + "000", b16="0b0000000000000001", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b1111111111111111", "0b0000000001000100", "0b0000000000000001")  # A=-1
        assert _cpu.evaluate(_in16="0b111" + "0" + "001100" + "100" + "000", b16="0b1111111111111111", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0000000000000001", "0b0000000001000101", "0b0000000000000001")  # A=D
        assert _cpu.evaluate(_in16="0b111" + "0" + "110000" + "100" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0001111111111111", "0b0", "0b0000000000000001", "0b0000000001000110", "0b0000000000000001")  # A=A
        assert _cpu.evaluate(_in16="0b111" + "1" + "110000" + "100" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0001111111111111", "0b0", "0b0001111111111111", "0b0000000001000111", "0b0000000000000001")  # A=M
        assert _cpu.evaluate(_in16="0b111" + "0" + "001101" + "100" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0001111111111111", "0b0", "0b1111111111111110", "0b0000000001001000", "0b0000000000000001")  # A=!D
        assert _cpu.evaluate(_in16="0b111" + "0" + "110001" + "100" + "000", b16="0b1110000000000000", reset="0b0") == \
            ("0b1110000000000000", "0b0", "0b0000000000000001", "0b0000000001001001", "0b0000000000000001")  # A=!A
        assert _cpu.evaluate(_in16="0b111" + "1" + "110001" + "100" + "000", b16="0b1110000000000000", reset="0b0") == \
            ("0b1110000000000000", "0b0", "0b0001111111111111", "0b0000000001001010", "0b0000000000000001")  # A=!M
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "100" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0001111111111111", "0b0", "0b0000000000000001", "0b0000000001001011", "0b0000000000000001")  # A=1
        assert _cpu.evaluate(_in16="0b111" + "1" + "001111" + "100" + "000", b16="0b0000000000000001", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b1111111111111111", "0b0000000001001100", "0b0000000000000001")  # A=-D
        assert _cpu.evaluate(_in16="0b111" + "0" + "110011" + "100" + "000", b16="0b1110000000000001", reset="0b0") == \
            ("0b1110000000000001", "0b0", "0b0000000000000001", "0b0000000001001101", "0b0000000000000001")  # A=-A
        assert _cpu.evaluate(_in16="0b111" + "1" + "110011" + "100" + "000", b16="0b1110000000000001", reset="0b0") == \
            ("0b1110000000000001", "0b0", "0b0001111111111111", "0b0000000001001110", "0b0000000000000001")  # A=-M
        assert _cpu.evaluate(_in16="0b111" + "0" + "011111" + "100" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0001111111111111", "0b0", "0b0000000000000010", "0b0000000001001111", "0b0000000000000001")  # A=D+1
        assert _cpu.evaluate(_in16="0b111" + "0" + "110111" + "100" + "000", b16="0b0010000000000000", reset="0b0") == \
            ("0b0010000000000000", "0b0", "0b0000000000000011", "0b0000000001010000", "0b0000000000000001")  # A=A+1
        assert _cpu.evaluate(_in16="0b111" + "1" + "110111" + "100" + "000", b16="0b0010000000000000", reset="0b0") == \
            ("0b0010000000000000", "0b0", "0b0010000000000001", "0b0000000001010001", "0b0000000000000001")  # A=M+1
        assert _cpu.evaluate(_in16="0b111" + "0" + "001110" + "100" + "000", b16="0b0010000000000001", reset="0b0") == \
            ("0b0010000000000001", "0b0", "0b0000000000000000", "0b0000000001010010", "0b0000000000000001")  # A=D-1
        assert _cpu.evaluate(_in16="0b111" + "0" + "110010" + "100" + "000", b16="0b0001111111111110", reset="0b0") == \
            ("0b0001111111111110", "0b0", "0b1111111111111111", "0b0000000001010011", "0b0000000000000001")  # A=A-1
        assert _cpu.evaluate(_in16="0b111" + "1" + "110010" + "100" + "000", b16="0b0001111111111110", reset="0b0") == \
            ("0b0001111111111110", "0b0", "0b0001111111111101", "0b0000000001010100", "0b0000000000000001")  # A=M-1
        assert _cpu.evaluate(_in16="0b111" + "0" + "000010" + "100" + "000", b16="0b0000000000000001", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0001111111111110", "0b0000000001010101", "0b0000000000000001")  # A=D+A
        assert _cpu.evaluate(_in16="0b111" + "1" + "000010" + "100" + "000", b16="0b0010000000000000", reset="0b0") == \
            ("0b0010000000000000", "0b0", "0b0010000000000001", "0b0000000001010110", "0b0000000000000001")  # A=D+M
        assert _cpu.evaluate(_in16="0b111" + "0" + "010011" + "100" + "000", b16="0b0010000000000001", reset="0b0") == \
            ("0b0010000000000001", "0b0", "0b1110000000000000", "0b0000000001010111", "0b0000000000000001")  # A=D-A
        assert _cpu.evaluate(_in16="0b111" + "1" + "010011" + "100" + "000", b16="0b1110000000000010", reset="0b0") == \
            ("0b1110000000000010", "0b0", "0b0001111111111111", "0b0000000001011000", "0b0000000000000001")  # A=D-M
        assert _cpu.evaluate(_in16="0b111" + "0" + "000111" + "100" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0001111111111111", "0b0", "0b0001111111111110", "0b0000000001011001", "0b0000000000000001")  # A=A-D
        assert _cpu.evaluate(_in16="0b111" + "1" + "000111" + "100" + "000", b16="0b0001111111111110", reset="0b0") == \
            ("0b0001111111111110", "0b0", "0b0001111111111101", "0b0000000001011010", "0b0000000000000001")  # A=M-D
        assert _cpu.evaluate(_in16="0b111" + "0" + "000000" + "100" + "000", b16="0b0001111111111101", reset="0b0") == \
            ("0b0001111111111101", "0b0", "0b0000000000000001", "0b0000000001011011", "0b0000000000000001")  # A=D&A
        assert _cpu.evaluate(_in16="0b111" + "1" + "000000" + "100" + "000", b16="0b0000000000000001", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0000000000000001", "0b0000000001011100", "0b0000000000000001")  # A=D&M
        assert _cpu.evaluate(_in16="0b111" + "0" + "010101" + "100" + "000", b16="0b0000000000000001", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0000000000000001", "0b0000000001011101", "0b0000000000000001")  # A=D|A
        assert _cpu.evaluate(_in16="0b111" + "1" + "010101" + "100" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0001111111111111", "0b0", "0b0001111111111111", "0b0000000001011110", "0b0000000000000001")  # A=D|M

        assert _cpu.evaluate(_in16="0b111" + "0" + "110000" + "010" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0001111111111111", "0b0", "0b0001111111111111", "0b0000000001011111", "0b0001111111111111")  # D=A
        assert _cpu.evaluate(_in16="0b111" + "0" + "110000" + "011" + "000", b16="0b0000000000000001", reset="0b0") == \
            ("0b0001111111111111", "0b1", "0b0001111111111111", "0b0000000001100000", "0b0001111111111111")  # MD=A
        assert _cpu.evaluate(_in16="0b111" + "0" + "101010" + "100" + "000", b16="0b0000000000000111", reset="0b0") == \
            ("0b0000000000000111", "0b0", "0b0000000000000000", "0b0000000001100001", "0b0001111111111111")  # A=0
        assert _cpu.evaluate(_in16="0b111" + "0" + "110000" + "101" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0000000000000000", "0b1", "0b0000000000000000", "0b0000000001100010", "0b0001111111111111")  # AM=A
        assert _cpu.evaluate(_in16="0b111" + "0" + "110000" + "110" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0001111111111111", "0b0", "0b0000000000000000", "0b0000000001100011", "0b0000000000000000")  # AD=A
        assert _cpu.evaluate(_in16="0b111" + "0" + "110000" + "111" + "000", b16="0b0001111111111111", reset="0b0") == \
            ("0b0000000000000000", "0b1", "0b0000000000000000", "0b0000000001100100", "0b0000000000000000")  # AMD=A

        # "JGT"="001", "JEQ"="010", "JGE"="011", "JLT"="100", "JNE"="101", "JLE"="110", "JMP"="111", None="000"
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000001100101",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "101010" + "000" + "111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000001111", "0b0000000000000000")  # 0;JMP
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010000",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "000" + "111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0000000000001111", "0b0000000000001111", "0b0000000000000000")  # 1;JMP
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010000",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "111010" + "000" + "111", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0000000000001111", "0b0000000000001111", "0b0000000000000000")  # -1;JMP

        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010000",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "101010" + "000" + "010", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000001111", "0b0000000000000000")  # 0;JEQ
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010000",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "000" + "010", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0000000000001111", "0b0000000000010001", "0b0000000000000000")  # 1;JEQ
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010010",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "111010" + "000" + "010", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0000000000001111", "0b0000000000010011", "0b0000000000000000")  # -1;JEQ

        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010100",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "101010" + "000" + "011", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000001111", "0b0000000000000000")  # 0;JGE
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010000",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "000" + "011", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0000000000001111", "0b0000000000001111", "0b0000000000000000")  # 1;JGE
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010000",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "111010" + "000" + "011", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0000000000001111", "0b0000000000010001", "0b0000000000000000")  # -1;JGE

        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010010",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "101010" + "000" + "100", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010011", "0b0000000000000000")  # 0;JLT
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010100",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "000" + "100", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0000000000001111", "0b0000000000010101", "0b0000000000000000")  # 1;JLT
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010110",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "111010" + "000" + "100", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0000000000001111", "0b0000000000001111", "0b0000000000000000")  # -1;JLT

        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010000",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "101010" + "000" + "101", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010001", "0b0000000000000000")  # 0;JNE
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010010",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "000" + "101", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0000000000001111", "0b0000000000001111", "0b0000000000000000")  # 1;JNE
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010000",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "111010" + "000" + "101", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0000000000001111", "0b0000000000001111", "0b0000000000000000")  # -1;JNE

        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010000",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "101010" + "000" + "110", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000001111", "0b0000000000000000")  # 0;JLE
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010000",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "000" + "110", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0000000000001111", "0b0000000000010001", "0b0000000000000000")  # 1;JLE
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010010",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "111010" + "000" + "110", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0000000000001111", "0b0000000000001111", "0b0000000000000000")  # -1;JLE

        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010000",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "101010" + "000" + "110", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000001111", "0b0000000000000000")  # 0;JLE
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010000",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "111111" + "000" + "110", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000001", "0b0", "0b0000000000001111", "0b0000000000010001", "0b0000000000000000")  # 1;JLE
        assert _cpu.evaluate(_in16="0b0000000000001111", b16="0b0000000000000000", reset="0b0") == \
            ("0b0000000000000000", "0b0", "0b0000000000001111", "0b0000000000010010",
             "0b0000000000000000")  # @0000000000001111
        assert _cpu.evaluate(_in16="0b111" + "0" + "111010" + "000" + "110", b16="0b0000000000000000", reset="0b0") == \
            ("0b1111111111111111", "0b0", "0b0000000000001111", "0b0000000000001111", "0b0000000000000000")  # -1;JLE

        # ROM32K
        assert _rom32k.evaluate(_in16="0b0000000000000000", addr15="0b000000000000000") == "0b0000000000000000"
        assert _rom32k.evaluate(_in16="0b0000000000000000", addr15="0b100000000000000") == "0b0000000000000000"
        _rom32k.rom_load = "0b1"  # enable ROM bootloader
        assert _rom32k.evaluate(_in16="0b1111111111111111", addr15="0b000000000000000") == "0b1111111111111111"
        assert _rom32k.evaluate(_in16="0b1111111111111111", addr15="0b100000000000000") == "0b1111111111111111"
        _rom32k.rom_load = "0b0"  # disable ROM bootloader
        assert _rom32k.evaluate(_in16="0b0000000000000000", addr15="0b000000000000000") == "0b1111111111111111"
        assert _rom32k.evaluate(_in16="0b0000000000000000", addr15="0b100000000000000") == "0b1111111111111111"

    else:
        # Computer
        if debug:
            '''
            "0b0000000000001111",  # @0000000000001111
            "0b1110111010010000",  # D=-1
            "0b1110111111001000",  # M=1
            "0b1110001101100000",  # A=!D
            "0b1110101010000111",  # 0;JMP
            '''
            _bin_filepaths = [r'python_hdl\debug.hack']

        else:
            # TODO: projects 1-11 accounted for, ASM included in assembler/interpreter
            _asm_filepaths = [
                r"nand2tetris\projects\04\fill\fill.hack",
                r"nand2tetris\projects\04\mult\mult.hack",
                r"nand2tetris\projects\06\add\add.hack",
                r"nand2tetris\projects\06\max\max.hack",
                r"nand2tetris\projects\06\max\maxL.hack",
                # r"nand2tetris\projects\06\pong\pong.hack",  # verified but slow
                # r"nand2tetris\projects\06\pong\pongL.hack",  # verified but slow
                r"nand2tetris\projects\06\rect\rect.hack",
                r"nand2tetris\projects\06\rect\rectL.hack",
                r"nand2tetris\projects\07\MemoryAccess\BasicTest\BasicTest.hack",
                r"nand2tetris\projects\07\MemoryAccess\PointerTest\PointerTest.hack",
                r"nand2tetris\projects\07\MemoryAccess\StaticTest\StaticTest.hack",
                r"nand2tetris\projects\07\StackArithmetic\SimpleAdd\SimpleAdd.hack",
                r"nand2tetris\projects\07\StackArithmetic\StackTest\StackTest.hack",
                r"nand2tetris\projects\08\FunctionCalls\FibonacciElement\FibonacciElement.hack",
                r"nand2tetris\projects\08\FunctionCalls\NestedCall\NestedCall.hack",
                r"nand2tetris\projects\08\FunctionCalls\SimpleFunction\SimpleFunction.hack",
                r"nand2tetris\projects\08\FunctionCalls\StaticsTest\StaticsTest.hack",
                r"nand2tetris\projects\08\ProgramFlow\BasicLoop\BasicLoop.hack",
                r"nand2tetris\projects\08\ProgramFlow\FibonacciSeries\FibonacciSeries.hack",

                # TODO: untested, still need project 12
                # r'nand2tetris\projects\09\Average\Average.hack',
                # r'nand2tetris\projects\09\Fraction\Fraction.hack',
                # r'nand2tetris\projects\09\HelloWorld\HelloWorld.hack',
                # r'nand2tetris\projects\09\List\List.hack',
                # # r'nand2tetris\projects\09\Square\Square.hack',  # too large, generates 17 bit addresses
                # r'nand2tetris\projects\10\ArrayTest\ArrayTest.hack',
                # # r'nand2tetris\projects\10\Square\Square.hack',  # too large, generates 17 bit addresses
                # r'nand2tetris\projects\11\Average\Average.hack',
                # # r'nand2tetris\projects\11\ComplexArrays\ComplexArrays.hack',  # too large, generates 17 bit addr
                # r'nand2tetris\projects\11\ConvertToBin\ConvertToBin.hack',
                # # r'nand2tetris\projects\11\Pong\Pong.hack',  # too large, generates 17 bit addresses
                # r'nand2tetris\projects\11\Seven\Seven.hack',
                # # r'nand2tetris\projects\11\Square\Square.hack',  # too large, generates 17 bit addresses
            ]

        for _bin_filepath in _bin_filepaths:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            with open(_bin_filepath) as _asm_file:
                program = _asm_file.readlines()

            computer = Computer(name="computer_main", debug=debug)
            print("%s %s: Loading ROM" % (current_time, _bin_filepath))
            computer.flash_rom(program)
            print("%s %s: Running program" % (current_time, _bin_filepath))
            for command in program:
                computer.evaluate()

            if debug:
                print()


if __name__ == "__main__":
    print("UNIT_TESTS: Initializing")
    main(test_all=True, debug=False)
    print("UNIT_TESTS: Complete!\n")

    print("COMPUTER_DEBUG: Initializing")
    main(test_all=False, debug=True)
    print("COMPUTER_DEBUG: Complete!\n")

    print("COMPUTER_ALL: Initializing")
    main(test_all=False, debug=False)
    print("COMPUTER_ALL: Complete!\n")
