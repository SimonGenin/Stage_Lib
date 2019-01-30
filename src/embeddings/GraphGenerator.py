import ast, asttokens
from collections import defaultdict

file = open("codes/code_1.py")
content = file.read()
file.close()

class AstGraphGenerator(object):

    def __init__(self, source):
        self.graph = dict()
        self.source = source  # lines of the source code

    def __str__(self):
        return str(self.graph)

    def _getid(self, node):
        try:
            lineno = node.lineno - 1
            return "%s: %s" % (type(node), self.source[lineno].strip())

        except AttributeError:
            return type(node)

    def load(self):
        return self.visit(ast.parse(self.source))

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        print('visit', item)
                        self.visit(item)

            elif isinstance(value, ast.AST):
                node_source = self._getid(node)
                value_source = self._getid(value)
                self.graph[node_source] = (value_source)
                self.graph[type(node)] = (type(value))
                self.visit(value)

x = AstGraphGenerator(content)
x.load()
# print(x.graph)