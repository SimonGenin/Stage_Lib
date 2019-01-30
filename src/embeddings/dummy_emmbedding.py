import plyplus, plyplus.grammars
from treelib import Tree


class PythonASTTreeBasedStructureGenerator():

    def __init__(self):
        self.code = None
        self.python_grammar = plyplus.Grammar(plyplus.grammars.open('python.g'))
        self.node_collection = None
        self.program = None
        self.tree = Tree()
        self.tree.create_node(data="program", identifier=0)
        self.has_been_generated = False

    def from_file(self, filepath):
        with open(filepath, "r") as file:
            self.code = file.read()
        return self

    def from_code(self, str_code):
        self.code = str_code
        return self

    def _generate(self):
        self._parse_code()
        self._fill_tree(self.program, 0, 0)
        self.has_been_generated = True

    def generate(self, as_copy=False):
        if not self.has_been_generated:
            self._generate()
        if as_copy:
            return Tree(self.tree)
        return self.tree

    def _parse_code(self):
        parsed_code = self.python_grammar.parse(self.code)
        tree_collection = parsed_code.select('*')

        # remove the end tokens
        new_list_of_nodes = []
        for i in tree_collection:
            if not isinstance(i, plyplus.plyplus.TokValue):
                new_list_of_nodes.append(i)

        self.node_collection = plyplus.strees.STreeCollection(new_list_of_nodes)
        self.program = self.node_collection[0]

    def _fill_tree(self, node, parent_id, current_id):

        # print("node is", node, " parent id is", parent_id, "and current id is", current_id)

        if isinstance(node, plyplus.plyplus.TokValue):
            # print("token", node, "so we return")
            return current_id

        sons = node.named_tail  # dict

        for son_key, son_value in sons.items():
            for son in son_value:  # son being Stree

                # we don't take the tokens (called False)
                if son_key == False: continue

                current_id += 1
                self.tree.create_node(data=son_key, parent=parent_id, identifier=current_id)
                current_id = self._fill_tree(son, current_id, current_id)

        return current_id

    def print_tree(self):
        if not self.has_been_generated:
            self._generate()
        print(self.tree.show())

    def print_ast_as_image(self, filename="AST.png"):
        if not self.has_been_generated:
            self._generate()
        self.program.to_png_with_pydot(filename)


gen = PythonASTTreeBasedStructureGenerator().from_file("../pytorch_extension/app.py")
gen.print_tree()
gen.print_ast_as_image()
