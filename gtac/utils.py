from __future__ import annotations

import copy
import json
import typing
import npn
import numpy as np
import os
import graphviz
import tempfile
import itertools
import time
import platform
import subprocess
import uuid
import pickle
import multiprocessing
import bitarray
import bitarray.util
import re

base_path = os.path.abspath(os.path.dirname(__file__))
abc_path = base_path + ("/bin/abc" if platform.system() == 'Linux' else "bin\\abc")
aigtoaig_path = base_path + ("/bin/aigtoaig" if platform.system() == 'Linux' else "bin\\aigtoaig")
temp_dir = None


def temp_dir_init():
    global temp_dir
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(dir=temp_dir)
        # print("aig file saved to %s" % temp_dir)
    return temp_dir


class Node:
    def __init__(self, var, left, right):
        self.var = var      # var >= 0: x_{var}; var = -1: constant 0; var = -2: all possibilities of input nodes (x_0, not(x_0), x_1, not(x_1), ...)
        self.left = left
        self.right = right
        self.input_symbol = None

    def is_leaf(self):
        return self.var is not None

    def __lt__(self, other):    # for sorting a list of nodes
        return id(self) < id(other)


class NodeWithInv(Node):
    def __init__(self, parent: Node, inverted, output_symbol=None):
        self._parent = parent
        self.inverted = inverted
        self.output_symbol = output_symbol

    @property
    def var(self):
        return self._parent.var

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        assert type(value) is Node
        self._parent = value

    @var.setter
    def var(self, value):
        self._parent.var = value

    @property
    def left(self):
        return self._parent.left

    @left.setter
    def left(self, value):
        self._parent.left = value

    @property
    def right(self):
        return self._parent.right

    @right.setter
    def right(self, value):
        self._parent.right = value

    @property
    def input_symbol(self):
        return self._parent.input_symbol

    @input_symbol.setter
    def input_symbol(self, value):
        self._parent.input_symbol = value


def plot_network(roots: NodeWithInv | Node | list[NodeWithInv | Node], view=False, tree=False, filename: str = None,
                 title: str = None, show_address=False, highlight_nodes: set = None):
    def plot_network_rec(node: NodeWithInv, parent: NodeWithInv | None, depth: int, g: graphviz.Digraph, plotted_nodes: set):
        # name = "x_%d" % node.var if node.var is not None else repr(node)
        if node is None:
            return
        if tree:
            node = copy.copy(node)                  # NodeWithInv
            node.parent = copy.copy(node.parent)    # Node
            new_nodes.append(node)  # save the node instance so that it won't be destroyed (to avoid id reallocation)
        name = str(id(node.parent))
        if node.is_leaf():
            g.attr('node', color='lightyellow' if highlight_nodes is None or node.parent not in highlight_nodes else 'yellow')
            g.node(name, (("x_%d" % node.var) if node.var >= 0 else "0" if node.var == -1 else "not determined") + ("\n" + hex(id(node.parent)) if show_address else ""))
            g.attr('node', color='lightgrey')
        else:
            g.attr('node', color='lightgrey' if highlight_nodes is None or node.parent not in highlight_nodes else 'yellow')
            g.node(name, "AND" + ("\n" + hex(id(node.parent)) if show_address else ""))
            g.attr('node', color='lightgrey')
            if node.parent not in plotted_nodes:
                plot_network_rec(node.left, node, depth + 1, g, plotted_nodes)
                plot_network_rec(node.right, node, depth + 1, g, plotted_nodes)
                plotted_nodes.add(node.parent)
        if parent is not None:
            if node.inverted:
                g.attr('edge', style='dashed')
            g.edge(name, str(id(parent.parent)))
            if node.inverted:
                g.attr('edge', style='')

    new_nodes = []
    g = graphviz.Digraph('logic_network' if title is None else title)

    g.attr('node', style='filled')
    if not isinstance(roots, list):
        roots = [roots]
    roots = [NodeWithInv(root, inverted=False) if type(root) is Node else root for root in roots]
    plotted_nodes = set()
    g.attr('node', color='lightgrey')
    for root in roots:
        plot_network_rec(root, None, 0, g, plotted_nodes)
    g.attr('node', color='lightblue')
    for i, root in enumerate(roots):
        name = str(id(root.parent)) + str(root.inverted)
        g.node(name, "o_%d" % i)
        if root.inverted:
            g.attr('edge', style='dashed')
        g.edge(str(id(root.parent)), name)
        if root.inverted:
            g.attr('edge', style='')
    if filename is not None:
        return g.render(outfile=filename, view=view)
    else:
        return g.render(directory=tempfile.mkdtemp(), view=view)


def compute_input_tt(num_inputs):
    input_tt = [bitarray.bitarray(2 ** num_inputs) for _ in range(num_inputs * 2)]
    for i in range(num_inputs):
        for j in range(2 ** num_inputs):
            input_tt[i * 2][j] = (j >> i) % 2
            input_tt[i * 2 + 1][j] = not input_tt[i * 2][j]
    return input_tt


def compute_tt(root: NodeWithInv | Node, num_inputs=None, input_tt=None, cache=None):
    def compute_tt_rec(root: NodeWithInv) -> bitarray.bitarray:
        if isinstance(cache, dict) and root in cache:
            return cache[root]
        if root is None:
            raise ValueError('root is None')
        elif root.is_leaf():
            if root.var >= 0:
                res = input_tt[root.var * 2]
            elif root.var == -1:    # -1 means constant node
                # print("Meet constant node.")
                res = bitarray.util.zeros(2 ** num_inputs)
            else:                   # -2 means not decided
                raise ValueError('root.value = %d, not supported' % root.var)
        elif isinstance(cache, dict) and root.parent in cache:
            res = cache[root.parent]
        else:
            left_tt = compute_tt_rec(root.left)
            right_tt = compute_tt_rec(root.right)
            res = left_tt & right_tt
        if root.inverted:
            res = ~res
        return res

    if input_tt is None:
        assert num_inputs is not None
        input_tt = compute_input_tt(num_inputs)
    elif num_inputs is None:
        num_inputs = base_2_log(len(input_tt[0]))
    if type(root) is Node:
        root = NodeWithInv(root, inverted=False)
    res = compute_tt_rec(root)
    return res


def compute_tts(roots: list[NodeWithInv], num_inputs=None, input_tt=None):
    if input_tt is None:
        assert num_inputs is not None
        input_tt = compute_input_tt(num_inputs)
    return [compute_tt(root, input_tt=input_tt) for root in roots]

def check_conflict(tree_stack: list[NodeWithInv], tt, input_tt, current_tt_conflict, cache, tolerance=0.0, eta=0.01):

    tt_size = len(tt)
    b = input_tt.copy()
    n = [bitarray.util.zeros(tt_size) for _ in input_tt]   # whether is unknown, 1: unknown, 0: known
    for node in reversed(tree_stack):
        if node.right is None:
            for i in range(len(input_tt)):
                n[i] = b[i] | n[i]              # 1 AND U = U
        else:
            left_tt = cache[node.left]          # all the left child have a known truth table
            for i in range(len(input_tt)):
                b[i] = b[i] & left_tt
                n[i] = n[i] & left_tt           # 0 AND U = 0
        if node.inverted:
            for i in range(len(input_tt)):
                b[i] = ~b[i]

    has_conflict = bitarray.bitarray(len(input_tt))
    # print("----------")
    for i in range(len(input_tt)):
        U_count = n[i].count()
        allowed_error = int(tolerance * tt_size) if tolerance is not None else 0
        if not n[i].any():  # if complete
            diff = (b[i] ^ tt) & ~n[i]
          
            for j in range(len(current_tt_conflict)):       # iterate among constructed output
                diff = diff | current_tt_conflict[j]
    
            error_count = diff.count()
            # print(error_count / tt_size)
            has_conflict[i] = (error_count > allowed_error) # and (error_count + U_count*eta > allowed_error)
        else:
            has_conflict[i] = False
    # print("----------")
    completeness = bitarray.bitarray([not n[i].any() for i in range(len(input_tt))])
    return has_conflict, completeness


def sequential_synthesis(roots, verbose=False, command='resyn2', title: str = "sequential"):
    temp_dir = temp_dir_init()
    sequential_filename = temp_dir + "/" + (title if title is not None else "")
    raw_filename = sequential_filename + "_raw"
    write_aiger(roots, raw_filename + ".aig", with_symbol_table=False)
    if command == 'deepsyn':
        while True:
            os.system(("%s %s \"&read %s; &deepsyn; &write -n %s;\"" %
                       (abc_path, "-c" if verbose else "-q", raw_filename + ".aig", sequential_filename + ".aig"))
                      + (" > NUL" if platform.system() == 'Windows' else ""))   # > /dev/null
            if os.path.exists(sequential_filename + ".aig"):
                break
            else:
                time.sleep(0.1)
    elif command == 'resyn2':
        while True:
            os.system(("%s %s \"read %s; resyn2; write %s;\"" %
                      (abc_path, "-c" if verbose else "-q", raw_filename + ".aig", sequential_filename + ".aig"))
                      + (" > NUL" if platform.system() == 'Windows' else ""))
            if os.path.exists(sequential_filename + ".aig"):
                break
            else:
                time.sleep(0.1)
    elif command == 'transtoch':
        while True:
            os.system(("%s %s \"&read %s; &transtoch -M 1 -V 0; &write -n %s;\"" %
                       (abc_path, "-c" if verbose else "-q", raw_filename + ".aig", sequential_filename + ".aig"))
                      + (" > NUL" if platform.system() == 'Windows' else ""))   # > /dev/null
            if os.path.exists(sequential_filename + ".aig"):
                break
            else:
                time.sleep(0.1)
    os.unlink(raw_filename + ".aag")
    os.unlink(raw_filename + ".aig")
    seq_roots = read_aiger(sequential_filename + ".aig")
    os.unlink(sequential_filename + ".aag")
    os.unlink(sequential_filename + ".aig")
    return seq_roots


def structural_hashing(roots, verbose=False):
    temp_dir = temp_dir_init()
    hashing_filename = temp_dir + "/hashing"
    raw_filename = hashing_filename + "_raw"
    write_aiger(roots, raw_filename + ".aig", with_symbol_table=False)
    os.system(("%s %s \"read %s; strash; write %s;\"" %
              (abc_path, "-c" if verbose else "-q", raw_filename + ".aig", hashing_filename + ".aig"))
              + (" > NUL" if platform.system() == 'Windows' else ""))
    return read_aiger(hashing_filename + ".aig")


def cec(roots_1, roots_2, verbose=0):
    temp_dir = temp_dir_init()
    if isinstance(roots_1, str):
        cec_filename_1 = roots_1
    else:
        cec_filename_1 = temp_dir + "/cec_1.aig"
        write_aiger(roots_1, cec_filename_1, with_symbol_table=False)
    if isinstance(roots_2, str):
        cec_filename_2 = roots_2
    else:
        cec_filename_2 = temp_dir + "/cec_2.aig"
        write_aiger(roots_2, cec_filename_2, with_symbol_table=False)
    res = subprocess.run([abc_path, "-c", "cec %s %s" % (cec_filename_1, cec_filename_2)], capture_output=True)
    stdout = res.stdout.decode("utf-8")
    equivalent = "Networks are equivalent" in stdout
    if verbose > 0 or not equivalent:
        print(stdout)
    return equivalent


def checkER(roots_1, roots_2):      # check error rate for circuits, roots_1 & roots_2
    '''Check the error rate.'''
    tt1 = compute_tts(roots_1, input_tt=compute_input_tt(8))
    tt2 = compute_tts(roots_2, input_tt=compute_input_tt(8))
    
    mismatch_mask = bitarray.util.zeros(len(tt1[0]))  # 256 bits
    for out1, out2 in zip(tt1, tt2):
        # print(out1)
        # print(out2)
        mismatch_mask |= (out1 ^ out2) 
    total_inputs = len(mismatch_mask)
    mismatch_count = mismatch_mask.count()
    error_rate = mismatch_count / total_inputs
    return error_rate

# def checkER(roots_1, roots_2):      # check error rate for circuits, roots_1 & roots_2
#     '''Check the error rate.'''
#     tt1 = compute_tts(roots_1, input_tt=compute_input_tt(8))
#     tt2 = compute_tts(roots_2, input_tt=compute_input_tt(8))
    
#     total_bits = 0
#     mismatch_bits = 0
    
#     for t1, t2 in zip(tt1, tt2):
#         total_bits += len(t1)
#         mismatch_bits += (t1 ^ t2).count()
    
#     error_rate = mismatch_bits / total_bits
#     return error_rate



def measureDelayArea(roots, verbose=False):
    '''Measure the delay'''
    temp_dir = temp_dir_init()
    delay_filename = temp_dir + "/delay_measurement"
    raw_filename = delay_filename + "_raw"
    library_path = "library/mcnc-aig.genlib"
    
    write_aiger(roots, raw_filename + ".aig", with_symbol_table=False)
    
    abc_commands = [
        f"read_aiger {raw_filename}.aig", 
        f"read_library {library_path}", 
        "map",                    
        "print_stats"              
    ]
    cmd_str = "; ".join(abc_commands)
    result = subprocess.run(
        ["abc", "-c", cmd_str],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    result = result.stdout + "\n" + result.stderr
    # print(f"ABC output for (first 300 chars):\n{result[:300]}...")
    if verbose:
        print("ABC command:", cmd)
        print("ABC output:", result.stdout)
        print("ABC error:", result.stderr)

    stats = {'power': 'N/A', 'delay': 'N/A', 'area': 'N/A', 'size': 'N/A'}
    
    patterns = {
        'area': r'area\s*=\s*([\d.]+)',
        'delay': r'delay\s*=\s*([\d.]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, result)
        if match:
            stats[key] = match.group(1)
    
    delay = stats['delay']
    area = stats['area']

    try:
        os.remove(raw_filename + ".aag")
        os.remove(raw_filename + ".aig")
    except FileNotFoundError:
        pass

    if delay is None or area is None:
        raise RuntimeError("Unable to parse from ABC output")

    return delay, area

def compute_critical_path(roots):
    '''Calculate critical path delay (logic depth)'''
    if not roots:
        return 0

    # Use visited set to avoid redundant calculations instead of caching depth
    visited = set()

    def traverse(node):
        # Handle case where node is None
        if node is None:
            return -1  # Return -1 to indicate invalid node

        # Leaf nodes (input variables)
        if node.is_leaf():
            return 0

        # Non-leaf nodes (AND gates)
        left_depth = traverse(node.left) if node.left else -1
        right_depth = traverse(node.right) if node.right else -1

        # Handle cases of missing child nodes
        if left_depth < 0 and right_depth < 0:
            depth = 0  # No valid child nodes
        elif left_depth < 0:
            depth = right_depth + 1  # Only right child node
        elif right_depth < 0:
            depth = left_depth + 1   # Only left child node
        else:
            depth = max(left_depth, right_depth) + 1

        return depth

    # Calculate maximum depth of all root nodes
    max_depth = 0
    for root in roots:
        if root:  # Ensure root node is valid
            depth = traverse(root)
            if depth > max_depth:
                max_depth = depth

    return max_depth

def read_aiger(filename=None, aiger_str: str = None) -> (list[NodeWithInv], dict):
    if filename is not None:
        filename_without_ext, file_ext = os.path.splitext(filename)
        if file_ext == '.aig':  # binary
            while True:
                os.system("%s %s.aig %s.aag" % (aigtoaig_path, filename_without_ext, filename_without_ext))
                if os.path.exists(filename_without_ext + ".aag"):
                    break
                else:
                    time.sleep(0.1)
        filename = filename_without_ext + ".aag"
        with open(filename) as f:
            aiger_str = f.read()
    f = aiger_str.split("\n")
    lineid = 0
    s = f[lineid]
    lineid += 1
    info = list(map(int, s.split(' ')[1:]))
    max_var_index, num_inputs, num_latches, num_outputs, num_ands = info
    assert num_latches == 0    # a boolean network without latches
    node_dict = {i: Node(None, None, None) for i in range(0, max_var_index + 1)}
    node_dict[0].var = -1   # constant zero
    input_indices, output_indices, output_not_list = [], [], []
    output_symbol_dict = {}
    for i in range(num_inputs):
        s = int(f[lineid])
        lineid += 1
        var_index = s // 2
        if var_index <= 0:
            print(aiger_str)
            raise ValueError()
        input_indices.append(var_index)
        node_dict[var_index].var = var_index - 1
        node_dict[var_index].left = None
        node_dict[var_index].right = None
    for i in range(num_outputs):
        s = int(f[lineid])
        lineid += 1
        output_indices.append(s // 2)
        output_not_list.append(s % 2 == 1)
    for i in range(num_ands):
        s, p, q = list(map(int, f[lineid].split(' ')))
        lineid += 1
        var_index = s // 2
        left_not = p % 2 == 1
        if left_not:
            left = NodeWithInv(node_dict[p // 2], True)
        else:
            left = NodeWithInv(node_dict[p // 2], False)
        right_not = q % 2 == 1
        if right_not:
            right = NodeWithInv(node_dict[q // 2], True)
        else:
            right = NodeWithInv(node_dict[q // 2], False)
        node_dict[var_index].var = None
        node_dict[var_index].left = left
        node_dict[var_index].right = right
    while f[lineid] != 'c' and f[lineid] != '' and lineid < len(f):
        id, symbol = f[lineid].split(' ')
        ioro, id = id[0], int(id[1:])
        if ioro == 'i':
            if input_indices[id] != 0:
                node_dict[input_indices[id]].input_symbol = symbol
        elif ioro == 'o':
            output_symbol_dict[id] = symbol
        lineid += 1
    output_nodes = []
    for i, (output_index, output_not) in enumerate(zip(output_indices, output_not_list)):
        output_nodes.append(NodeWithInv(node_dict[output_index], output_not,
                                        output_symbol_dict[i] if i in output_symbol_dict else None))
    if num_outputs == 1:    # boolean function
        output_nodes = output_nodes[0]
    return output_nodes, info


def write_aiger(root: NodeWithInv | list[NodeWithInv], filename: str = None, with_symbol_table=True, num_inputs=None) -> str:
    def write_aiger_inputs_rec(root: NodeWithInv, info: dict, visited_nodes: set):
        if root.is_leaf():
            # root.id = root.var + 1  # id starts from 1
            info['node_to_id'][root.parent] = root.var + 1
            if root.var != -1:
                info['inputs'][root.var + 1] = root.input_symbol
        else:
            if root not in visited_nodes:
                write_aiger_inputs_rec(root.left, info, visited_nodes)
                write_aiger_inputs_rec(root.right, info, visited_nodes)
                visited_nodes.add(root)

    def write_aiger_rec(root: NodeWithInv, info: dict):
        if not root.is_leaf():
            if root.parent not in info['node_to_id']:
                left_id, left_not = write_aiger_rec(root.left, info)
                right_id, right_not = write_aiger_rec(root.right, info)
                info['node_to_id'][root.parent] = info['node_id']
                info['ands_detail'].append((info['node_id'], left_id * 2 + left_not, right_id * 2 + right_not))
                info['node_id'] += 1
        # assert root.id is not None
        return info['node_to_id'][root.parent], root.inverted

    info = {'node_id': 1, 'ands_detail': [], 'inputs': dict(), 'node_to_id': dict()}
    # add constant nodes
    constant_zero = NodeWithInv(Node(-1,None,None), 0) 
    constant_one = NodeWithInv(Node(-1,None,None), 1) 
    info['node_to_id'][constant_zero] = 0
    info['node_to_id'][constant_one] = 1

    if isinstance(root, NodeWithInv):
        root = [root]
    root = root.copy()
    for i, r in enumerate(root):
        if type(r) is Node:
            root[i] = NodeWithInv(r, inverted=False)
    for r in root:
        write_aiger_inputs_rec(r, info, set())
    if num_inputs is None:
        if len(info['inputs'].keys()) > 0:
            info['node_id'] = max(info['inputs'].keys()) + 1   # skip the input ids
        else:
            info['node_id'] = 1     # only have constant nodes
    else:
        if max(info['inputs'].keys()) > num_inputs:
            print("Warning: num_inputs is smaller than the actual number of inputs, use the latter instead")
            num_inputs = max(info['inputs'].keys())
        info['node_id'] = num_inputs + 1    # input ids from 1 to num_inputs
    outputs = []
    for r in root:
        output_id, output_not = write_aiger_rec(r, info)
        outputs.append(output_id * 2 + output_not)
    info['node_id'] -= 1
    aiger_str = "aag %d %d 0 %d %d\n" % \
                (info['node_id'], len(info['inputs']), len(outputs), len(info['ands_detail']))
    for i in sorted(info['inputs'].keys()):
        if i not in [0]:       # jump constant nodes when writing input list
            aiger_str += str(i * 2) + '\n'
    for i in outputs:
        aiger_str += str(i) + '\n'
    for and_id, left_literal, right_literal in info['ands_detail']:
        aiger_str += "%d %d %d\n" % (and_id * 2, left_literal, right_literal)
    if with_symbol_table:
        for i, (id, symbol) in enumerate(sorted(info['inputs'].items(), key=lambda x: x[0])):
            if symbol is not None:
                aiger_str += 'i%d %s\n' % (i, symbol)
        # output_used_symbols = set()
        for i, r in enumerate(root):
            if r.output_symbol is not None:
                aiger_str += 'o%d %s\n' % (i, r.output_symbol)

    if filename is not None:
        filename_without_ext, file_ext = os.path.splitext(filename)
        with open(filename_without_ext + ".aag", 'w', newline='\n') as f:
            f.write(aiger_str)
        if file_ext == '.aig':     # binary
            while True:
                os.system("%s %s.aag %s.aig" % (aigtoaig_path, filename_without_ext, filename_without_ext))
                if os.path.exists(filename_without_ext + ".aig"):
                    break
                else:
                    time.sleep(0.1)
            # os.unlink(filename_without_ext + ".aag")
    return aiger_str


def get_aig_info(filename):
    if not os.path.exists(filename):
        return [-1, -1, -1, -1, -1]
    with open(filename, 'rb') as f:
        s, c = "", ""
        while c != '\n':
            s += c
            c = f.read(1).decode('ascii')
        info = list(map(int, s.split(' ')[1:]))
    return info


def npn_transform(roots: NodeWithInv | list[NodeWithInv], phase, perm, output_invs: bool | list[bool]):
    def n_transform(root: NodeWithInv, phase, perm, transformed_node: set, transformed_node_with_inv: set):
        if root.is_leaf():
            if root not in transformed_node_with_inv:
                if phase[root.var] > 0 and root.var > -1:       
                    root.inverted = not root.inverted
                    transformed_node_with_inv.add(root)
            if root.parent not in transformed_node:
                # root.var = perm[root.var]
                transformed_node.add(root.parent)
        else:
            n_transform(root.left, phase, perm, transformed_node, transformed_node_with_inv)
            n_transform(root.right, phase, perm, transformed_node, transformed_node_with_inv)

    leaf_nodes = set()
    transformed_node_with_inv = set()
    if isinstance(roots, NodeWithInv):
        roots = [roots]
    for root in roots:
        n_transform(root, phase, perm, leaf_nodes, transformed_node_with_inv)
    for node in leaf_nodes:
        if node.var >= 0:   # only permutate on not constant nodes
            node.var = perm[node.var]
    if isinstance(output_invs, bool):
        output_invs = [output_invs]
    for output_inv, root in zip(output_invs, roots):
        if output_inv:
            root.inverted = not root.inverted
    return roots if len(roots) > 1 else roots[0]


def npn_transform_tt(tt: bitarray.bitarray, phase, perm, output_invs: bool | list[bool]):
    return bitarray.bitarray(npn.transform_tt(tt.tolist(), phase, perm, output_invs))


def tt_to_str(tt, hex=False):
    if hex:
        return ("%0" + str(len(tt) // 4) + "x") % npn.tt_to_int(tt)
    else:
        return "".join("1" if i else "0" for i in tt)


def count_num_ands(root: NodeWithInv | list | None, node_set: set = None):
    if node_set is None:
        node_set = set()
    if isinstance(root, list):
        return sum([count_num_ands(r, node_set) for r in root])
    if root is None or root.var is not None or root.parent in node_set:
        return 0
    else:
        node_set.add(root.parent)
        return count_num_ands(root.left, node_set) + count_num_ands(root.right, node_set) + 1


def count_seq_length(root: NodeWithInv | list[NodeWithInv] | Node | list[Node],
                     maximum_length=None, cut=None, cumulative_length=0):
    if isinstance(root, list):
        for r in root:
            if type(r) is Node:
                r = NodeWithInv(r, inverted=False)
            cumulative_length = count_seq_length(r, maximum_length, cut, cumulative_length)
            if cumulative_length == -1 or cumulative_length > maximum_length:
                return -1
        return cumulative_length
    if root is None or root.is_leaf() or (cut is not None and root.parent in cut):
        return cumulative_length + 1
    else:
        cumulative_length = count_seq_length(root.left, maximum_length, cut, cumulative_length)
        if cumulative_length == -1 or cumulative_length > maximum_length:
            return -1
        cumulative_length = count_seq_length(root.right, maximum_length, cut, cumulative_length)
        if cumulative_length == -1 or cumulative_length + 1 > maximum_length:
            return -1
        return cumulative_length + 1


def check_integrity(root: NodeWithInv | list[NodeWithInv] | None):
    if isinstance(root, list):
        return all([check_integrity(r) for r in root])
    if root is None:
        return False
    elif root.is_leaf():
        return True
    else:
        return check_integrity(root.left) and check_integrity(root.right)


def base_2_log(n: int):     # see src/misc/util/abc_global.h in abc
    if n < 2:
        return n
    else:
        r = 0
        n -= 1
        while n > 0:
            r += 1
            n >>= 1
        return r


def get_inputs_rec(node: NodeWithInv | list[NodeWithInv], visited=None, inputs_dict=None):
    if visited is None:
        visited = set()
    if inputs_dict is None:
        inputs_dict = dict()
    if isinstance(node, list):
        for node_i in node:
            get_inputs_rec(node_i, visited, inputs_dict)
        return inputs_dict
    if node in visited:
        return inputs_dict
    visited.add(node)
    if node.is_leaf():
        if node.var not in inputs_dict:
            inputs_dict[node.var] = set()
        inputs_dict[node.var].add(node)
    else:
        get_inputs_rec(node.left, visited, inputs_dict)
        get_inputs_rec(node.right, visited, inputs_dict)
    return inputs_dict


def popcount(tt, var_id: int):
    popcount_0, popcount_1 = 0, 0
    interval = 1 << var_id
    for i, t in enumerate(tt):
        if t:
            if i & interval == 0:
                popcount_0 += 1
            else:
                popcount_1 += 1
    return popcount_0, popcount_1


def reverse_permutation(perm: list):
    reversed_perm = copy.copy(perm)
    for i, p in enumerate(perm):
        reversed_perm[p] = i
    return reversed_perm


def NPNP_transformation(nodes: list[NodeWithInv], npnp_info: list):
    new_nodes = copy.deepcopy(nodes)
    input_flip, input_perm, output_flip, output_perm = npnp_info
    # output perm
    new_nodes = [new_nodes[i] for i in reverse_permutation(output_perm)]
    # input flip
    inputs_dict = get_inputs_rec(new_nodes)
    for i, f in enumerate(input_flip):
        if f and i in inputs_dict:
            for node_with_inv in inputs_dict[i]:
                node_with_inv.inverted = not node_with_inv.inverted
    # input perm
    for i, p in enumerate(reverse_permutation(input_perm)):
        if p in inputs_dict:
            node = list(inputs_dict[p])[0].parent
            node.var = i
    # output flip
    for new_node, f in zip(new_nodes, output_flip):
        if f:
            new_node.inverted = not new_node.inverted
    return new_nodes


def detect_circle(root: NodeWithInv | list[NodeWithInv], visiting=None, visited=None):
    if visiting is None or visited is None:
        visiting, visited = set(), set()
    if isinstance(root, list):
        return any([detect_circle(r, visiting, visited) for r in root])
    if root.parent in visiting and root.parent not in visited:
        return True
    if root.parent in visited:
        return False
    visiting.add(root.parent)
    if not root.is_leaf():
        res = detect_circle(root.left, visiting, visited) or detect_circle(root.right, visiting, visited)
    else:
        res = False
    visited.add(root.parent)
    return res


def get_subcircuit_nodes(inputs, outputs):
    def get_subcircuit_nodes_rec(root: NodeWithInv, inputs, node_set):
        if root.parent not in node_set:
            node_set.add(root.parent)
            if not root.is_leaf() and root.parent not in inputs:
                get_subcircuit_nodes_rec(root.left, inputs, node_set)
                get_subcircuit_nodes_rec(root.right, inputs, node_set)
    node_set = set()
    for output in outputs:
        if type(output) is Node:
            output = NodeWithInv(output, False)
        get_subcircuit_nodes_rec(output, inputs, node_set)
    return node_set


def aag_to_aig(aag_path, aig_path):
    os.system("%s %s %s" % (aigtoaig_path, aag_path, aig_path))


# https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


# https://pypi.org/project/nvidia-ml-py/
def gpu_count():
    from pynvml import nvmlInit, nvmlDeviceGetCount
    nvmlInit()
    return nvmlDeviceGetCount()

def _mp_dataset_run(num_processes, mapping_func, train_data, chunk_size, q):
    print("process start running")
    pool = multiprocessing.Pool(num_processes)
    chunk_id = 0
    while True:
        if q.qsize() < chunk_size:
            for mapped_data in pool.imap_unordered(mapping_func, train_data[chunk_id * chunk_size: (chunk_id + 1) * chunk_size], chunksize=128):
                if not isinstance(mapped_data, Exception):
                    q.put(mapped_data)
            chunk_id += 1
            if chunk_id * chunk_size >= len(train_data):
                chunk_id = 0
        else:
            time.sleep(0.1)

class MPDataset:
    def __init__(self,
                 data: list,
                 mapping_func: typing.Callable,
                 kwargs: dict = None,
                 num_processes=1,
                 validation_split=0.1):
        self.train_data, self.validation_data = \
            data[: int(len(data) * (1 - validation_split))], data[int(len(data) * (1 - validation_split)):]
        self.mapping_func = mapping_func if kwargs is None else lambda x: mapping_func(x, **kwargs)
        self.num_processes = num_processes
        self.queue = multiprocessing.Queue()
        # self.pool = multiprocessing.Pool(num_processes)
        self.chunk_size = 128 * 8

        
        if self.num_processes > 1:
            self.process = multiprocessing.Process(
                target=_mp_dataset_run,
                args=(num_processes, mapping_func, self.train_data, self.chunk_size, self.queue)
            )
            self.process.start()

    def __len__(self):
        return len(self.train_data)

    def train_generator(self):
        counter = 0
        if self.num_processes > 1:
            while True:
                if not self.queue.empty():
                    data = self.queue.get()
                    if data is not None and not isinstance(data, Exception):        # Modified: Jump invalid data.
                        yield data
                        counter += 1
                    if counter >= len(self.train_data):
                        break
        else:
            for data in self.train_data:
                result = self.mapping_func(data)
                if result is not None and not isinstance(result, Exception):
                    yield result

    def validation_generator(self):
        for data in self.validation_data:
            mapped_data = self.mapping_func(data)
            if mapped_data is not None and not isinstance(mapped_data, Exception):      # Modified: Jump invalid data.
                yield mapped_data

    def benchmark(self, num_data=512):
        counter = 0
        start_time = time.time()
        for data in self.train_generator():
            counter += 1
            if counter >= num_data:
                break
        total_time = time.time() - start_time
        print("%d data elements processed in %.2fs, average %.4fs per data element" %
              (num_data, total_time, total_time / num_data))


class LogWrapper(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately

    def flush(self) :
        for f in self.files:
            f.flush()