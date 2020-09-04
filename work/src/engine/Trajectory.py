class _StateHistory(object):

    def __init__(self, num_entries, q_shape, p_shape):
        self._q_history = torch.zeros((num_entries, *q_shape))
        self._p_history = torch.zeros((num_entries, *p_shape))
        self._t_history = torch.zeros(num_entries)
        self._H_history = torch.zeros(num_entries)
        self._num_entries = num_entries
        self._entry_count = 0

    def __len__(self):
        return self._entry_count

    def __getitem__(self, index):
        if type(index) is not tuple:
            return (self._q_history[index], self._p_history[index],
                    self._t_history[index], self._H_history[index])
        elif len(index) == 2:
            return (self._q_history[index[0]], self._p_history[index[0]],
                    self._t_history[index[0]], self._H_history[index[0]])[index[1]]
        else:
            return (self._q_history[(index[0], *index[2:])], self._p_history[(index[0], *index[2:])],
                    self._t_history[index[0]], self._H_history[index[0]])[index[1]]

    def _append(self, q, p, t, H=None):
        if self._entry_count == self._num_entries:
            raise Exception('History Overflow')
        self._q_history[self._entry_count] = q.detach().clone()
        self._p_history[self._entry_count] = p.detach().clone()
        self._t_history[self._entry_count] = t
        if H is not None:
            self._H_history[self._entry_count] = H
        self._entry_count += 1