"""Scoped repairs to the pinned native HGGD epoch functions."""
import ast


def compile_functions(module, backward, shift, shift_min_labels):
    """Keep native geometry/supervision; replace accumulation and scalar logging."""
    from pathlib import Path
    source = Path(module.__file__).read_text()
    tree = ast.parse(source, filename=module.__file__)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('train', 'validate')]
    counts = {'backward': 0, 'step': 0, 'alias': 0, 'shift': 0}

    class Repair(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value) == 'loss.backward()':
                counts['backward'] += 1
                return ast.parse('_backward(loss, epoch)').body[0]
            return self.generic_visit(node)

        def visit_If(self, node):
            condition = ast.unparse(node.test)
            if condition == 'batch_idx > 0 and batch_idx % args.step_cnt == 0':
                counts['step'] += 1
                return None
            if condition == 'len(cur_labels) > 1000000.0':
                counts['shift'] += 1
                node.test = ast.parse('len(cur_labels) >= _shift_min_labels', mode='eval').body
            if condition in ('len(rect_ggs) == 0', 'pc_group.shape[0] == 0'):
                # Epoch scheduling requires an explicit update for each batch.
                # Never silently skip a failed training sample or invent labels.
                for index, statement in enumerate(node.body):
                    if isinstance(statement, ast.Continue):
                        node.body[index] = ast.parse("raise ValueError('HGGD produced no local training proposals; initialize from trained weights or increase anchor pretraining epochs')").body[0]
            return self.generic_visit(node)

        def visit_AugAssign(self, node):
            if ast.unparse(node.target) == 'loss':
                counts['alias'] += 1
                return ast.Assign(targets=[node.target], value=ast.BinOp(left=ast.Name('loss', ast.Load()), op=node.op, right=node.value))
            if ast.unparse(node.target).startswith('sum_'):
                node.value = ast.Call(func=ast.Name('_scalar', ast.Load()), args=[node.value], keywords=[])
            return self.generic_visit(node)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'item' and not node.args and not node.keywords:
                return ast.Call(func=ast.Name('_scalar', ast.Load()), args=[node.func.value], keywords=[])
            return self.generic_visit(node)

    # Validation retains its native handling of empty detections.
    train = next(n for n in functions if n.name == 'train')
    train = Repair().visit(train)
    if counts != {'backward': 1, 'step': 1, 'alias': 1, 'shift': 1}:
        raise ValueError('Pinned HGGD epoch driver contract changed: ' + str(counts))
    valid = next(n for n in functions if n.name == 'validate')
    class ScalarValidation(ast.NodeTransformer):
        visit_Call = Repair.visit_Call
        def visit_AugAssign(self, node):
            if isinstance(node.op, ast.Div) and ast.unparse(node.target) == "results['losses'][ln]":
                return ast.parse("results['losses'][ln] = results['losses'].get(ln, 0.) / batch_idx").body[0]
            return self.generic_visit(node)
    valid = ScalarValidation().visit(valid)
    program = ast.fix_missing_locations(ast.Module(body=[train, valid], type_ignores=[]))
    def scalar(value):
        return float(value.detach()) if hasattr(value, 'detach') else float(value)
    namespace = dict(vars(module), _backward=backward, _scalar=scalar,
                     shift_anchors=shift, _shift_min_labels=shift_min_labels)
    exec(compile(program, module.__file__, 'exec'), namespace)
    return namespace['train'], namespace['validate']
