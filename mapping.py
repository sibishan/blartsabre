from bidict import bidict

class Mapping(bidict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def l_to_p(self, l_idx):
        """Get physical qubit index from logical qubit index"""
        return self[l_idx]
    
    def p_to_l(self, p_idx):
        """Get logical qubit index from physical qubit index"""
        return self.inv[p_idx]

    def swap_p_qubits(self, p_q1, p_q2):
        temp1 = self.inv[p_q1]
        temp2 = self.inv[p_q2]
        self.inv[p_q1] = None
        self.inv[p_q2] = temp1
        self.inv[p_q1] = temp2
    
    def get_free_p_nodes(self):
        return [self.l_to_p(i) for i in self if i < 0]

