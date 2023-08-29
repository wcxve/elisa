import ast

class VariableVisitor(ast.NodeVisitor):
    def __init__(self):
        self.variables = []

    def visit_Name(self, node):
        self.variables.append(node.id)

def find_variables(expression):
    # 解析表达式
    tree = ast.parse(expression, mode='eval')

    # 遍历AST并找到变量
    visitor = VariableVisitor()
    visitor.visit(tree)

    return visitor.variables

from sympy import expand, MatrixSymbol
a=MatrixSymbol('a', 1, 1)
b=MatrixSymbol('b', 1, 1)
c=MatrixSymbol('c', 1, 1)
str(expand(c*(a+b)))