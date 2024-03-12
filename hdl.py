""""
A rough example of a Python implementation of the Nand2Tetris HDL

NAND is a primitive implemented at the hardware level so need to define the logic ourselves
All subsequent gates can be expressed via increasingly complex abstractions of NAND
This is typically provided to the student as a built-in class for Nand2Tetris
All subsequent gates must be implemented by the student
"""

# TODO: gates can be passed function inputs that do nothing


class Gate(object):
    def __init__(self):
        # Gates values should only be initialized at runtime
        self.a = None
        self.b = None
        self.sel = None
        self.sel2 = None
        self._in = None

    def evaluate(self, a=None, b=None, sel=None, sel2=None, _in=None):
        """
        validate input, None = uninitialized
        """
        if a is not None:
            if type(a) is not str:
                a = bin(a)
            if a not in ("0b0", "0b1"):
                raise RuntimeError("a input must be 1 bit (0/1)")
            self.a = a

        if b is not None:
            if type(b) is not str:
                b = bin(b)
            if b not in ("0b0", "0b1"):
                raise RuntimeError("b input must be 1 bit (0/1)")
            self.b = b

        if sel is not None:
            if type(sel) is not str:
                sel = bin(sel)
            if sel not in ("0b0", "0b1"):
                raise RuntimeError("sel input must be 1 bit (0/1)")
            self.sel = sel

        if sel2 is not None:
            if type(sel2) is not str:
                sel2 = bin(sel2)
            if sel2 not in ("0b00", "0b01", "0b10", "0b11"):
                raise RuntimeError("sel2 input must be 2 bits (0/1*2)")
            self.sel2 = sel2
            
        if _in is not None:
            if type(_in) is not str:
                _in = bin(_in)
            if _in not in ("0b0", "0b1"):
                raise RuntimeError("_in input must be 1 bit (0/1)")
            self._in = _in

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
        # this should be the only point a python built-in is used for logic
        if int(self.a, 2) and int(self.b, 2):
            return "0b0"
        else:
            return "0b1"


class NotGate(Gate):
    """
    For a single input, return the opposite
    """
    def calculate(self):
        """
        A single input/output can still have a one-to-many relationship with other gates
        In this case it is passed to both inputs of the NAND gate
        """
        return NandGate().evaluate(a=self.a, b=self.a)


class AndGate(Gate):
    """
    For two 1 inputs return a 1 output, else return a 0 output
    """
    def calculate(self):
        """
        Now we can combine a NAND with a NOT to get a regular AND gate
        """
        nand_a = NandGate().evaluate(a=self.a, b=self.b)
        return NotGate().evaluate(a=nand_a)


class OrGate(Gate):
    """
    If either of the two inputs are 1 return a 1 output, else return a 0 output
    """
    def calculate(self):
        nand_a = NandGate().evaluate(a=self.a, b=self.a)
        nand_b = NandGate().evaluate(a=self.b, b=self.b)
        return NandGate().evaluate(a=nand_a, b=nand_b)


class XorGate(Gate):
    """
    If the two inputs are different return a 1 output, else return a 0 output
    """
    def calculate(self):
        nand_a = NandGate().evaluate(a=self.a, b=self.b)
        nand_b = NandGate().evaluate(a=self.a, b=nand_a)
        nand_c = NandGate().evaluate(a=nand_a, b=self.b)
        return NandGate().evaluate(a=nand_b, b=nand_c)


class NorGate(Gate):
    """
    If either of the two inputs are 1 return a 0 output, else return a 1 output
    """
    def calculate(self):
        raise NotImplementedError


class XNorGate(Gate):
    """
    If the two inputs are the same return a 1 output, else return a 0 output
    """
    def calculate(self):
        raise NotImplementedError


class Mux(Gate):
    """
    Select an output from two inputs, only chosen input will be emitted
    """
    def calculate(self):
        nand_a = NandGate().evaluate(a=self.sel, b=self.sel)
        nand_b = NandGate().evaluate(a=self.b, b=self.sel)
        nand_c = NandGate().evaluate(a=nand_a, b=self.a)
        return NandGate().evaluate(a=nand_b, b=nand_c)


class DMux(Mux):
    """
    Select one of two outputs, input passes through and unselected output is always 0
    """
    def calculate(self):
        nand_a = NandGate().evaluate(a=self.sel, b=self.sel)
        nand_b = NandGate().evaluate(a=self._in, b=nand_a)
        nand_c = NandGate().evaluate(a=self.sel, b=self._in)
        return NandGate().evaluate(a=nand_b, b=nand_b), NandGate().evaluate(a=nand_c, b=nand_c)


class DMux4Way(DMux):
    """
    With a 2 bit selector choose one of four outputs, input passes through and unselected is always 0
    """
    def calculate(self):
        # note: HDL & Python have opposite endianess
        # HDL: sel = 2 = 01 // LSB first = little endian
        # Python: sel2 = 2 = 0b10 // MSB first = big endian
        # sel[0] == sel2[-1] // sel[-1] == sel2[2+0]
        dmux_0 = DMux().evaluate(_in=self._in, sel="0b"+self.sel2[2])
        dmux_1 = DMux().evaluate(_in=dmux_0[0], sel="0b"+self.sel2[3])
        dmux_2 = DMux().evaluate(_in=dmux_0[1], sel="0b"+self.sel2[3])
        return dmux_1[0], dmux_1[1], dmux_2[0], dmux_2[1]


def main():
    """
    Sanity check our truth tables for each gate as implemented
    These unit tests are typically provided to the student in Nand2Tetris so they can confirm their results
    """
    _nand = NandGate()
    _not = NotGate()
    _and = AndGate()
    _or = OrGate()
    _xor = XorGate()
    # TODO: nor, xnor
    _mux = Mux()
    _dmux = DMux()
    _dmux4way = DMux4Way()

    # input size unit test
    runtime_errors = 0
    for i in range(0, 7):
        if i <= 1:
            test = _nand.evaluate(a="0b0", b="0b0", sel="0b0", sel2="0b00", _in="0b0")  # pass
            test = _nand.evaluate(a="0b1", b="0b1", sel="0b1", sel2="0b11", _in="0b1")  # pass
        try:
            if i == 2:
                test = _nand.evaluate(a="0b10", b="0b0", sel="0b0", sel2="0b00", _in="0b0")  # fail
            if i == 3:
                test = _nand.evaluate(a="0b0", b="0b10", sel="0b0", sel2="0b00", _in="0b0")  # fail
            if i == 4:
                test = _nand.evaluate(a="0b0", b="0b0", sel="0b10", sel2="0b00", _in="0b0")  # fail
            if i == 5:
                test = _nand.evaluate(a="0b0", b="0b0", sel="0b0", sel2="0b00", _in="0b10")  # fail
            if i == 6:
                test = _nand.evaluate(a="0b0", b="0b0", sel="0b0", sel2="0b100", _in="0b0")  # fail
        except RuntimeError:
            runtime_errors += 1
    assert runtime_errors == 5

    # For two 1 inputs return a 1 output, else return a 1 output
    assert _nand.evaluate(a="0b1", b="0b1") == "0b0"
    assert _nand.evaluate(a="0b1", b="0b0") == "0b1"
    assert _nand.evaluate(a="0b0", b="0b1") == "0b1"
    assert _nand.evaluate(a="0b0", b="0b0") == "0b1"

    # For a single input, return the opposite
    assert _not.evaluate(a="0b1") == "0b0"
    assert _not.evaluate(a="0b0") == "0b1"

    # For two 1 inputs return a 1 output, else return a 0 output
    assert _and.evaluate(a="0b1", b="0b1") == "0b1"
    assert _and.evaluate(a="0b1", b="0b0") == "0b0"
    assert _and.evaluate(a="0b0", b="0b1") == "0b0"
    assert _and.evaluate(a="0b0", b="0b0") == "0b0"

    # If either of the two inputs are 1 return a 1 output, else return a 0 output
    assert _or.evaluate(a="0b1", b="0b1") == "0b1"
    assert _or.evaluate(a="0b1", b="0b0") == "0b1"
    assert _or.evaluate(a="0b0", b="0b1") == "0b1"
    assert _or.evaluate(a="0b0", b="0b0") == "0b0"

    # If the two inputs are different return a 1 output, else return a 0 output
    assert _xor.evaluate(a="0b1", b="0b1") == "0b0"
    assert _xor.evaluate(a="0b1", b="0b0") == "0b1"
    assert _xor.evaluate(a="0b0", b="0b1") == "0b1"
    assert _xor.evaluate(a="0b0", b="0b0") == "0b0"

    # Select an output from two inputs, only chosen input will be emitted
    assert _mux.evaluate(a="0b1", b="0b0", sel="0b0") == "0b1"
    assert _mux.evaluate(a="0b1", b="0b0", sel="0b1") == "0b0"
    assert _mux.evaluate(a="0b0", b="0b1", sel="0b0") == "0b0"
    assert _mux.evaluate(a="0b0", b="0b1", sel="0b1") == "0b1"
    
    # Select one of two outputs, input passes through and unselected output is always 0
    assert _dmux.evaluate(_in="0b0", sel="0b0") == ("0b0", "0b0")
    assert _dmux.evaluate(_in="0b0", sel="0b1") == ("0b0", "0b0")
    assert _dmux.evaluate(_in="0b1", sel="0b0") == ("0b1", "0b0")
    assert _dmux.evaluate(_in="0b1", sel="0b1") == ("0b0", "0b1")

    # With a 2 bit selector choose one of four outputs, input passes through and unselected is always 0
    assert _dmux4way.evaluate(_in="0b0", sel2="0b00") == ("0b0", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b0", sel2="0b01") == ("0b0", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b0", sel2="0b10") == ("0b0", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b0", sel2="0b11") == ("0b0", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b1", sel2="0b00") == ("0b1", "0b0", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b1", sel2="0b01") == ("0b0", "0b1", "0b0", "0b0")
    assert _dmux4way.evaluate(_in="0b1", sel2="0b10") == ("0b0", "0b0", "0b1", "0b0")
    assert _dmux4way.evaluate(_in="0b1", sel2="0b11") == ("0b0", "0b0", "0b0", "0b1")


if __name__ == "__main__":
    main()
