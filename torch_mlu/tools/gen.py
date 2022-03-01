# Generates parts of C++ header/source files:
# csrc/generated/aten_mlu_type_default.cpp
# csrc/generated/aten_mlu_type_default.h
# csrc/generated/autograd/functions.cpp
# csrc/generated/autograd/functions.h
# csrc/aten/aten_mlu_type.cpp
# csrc/aten/aten_mlu_custom_type.cpp
# csrc/aten/aten_mlu_type.h
# csrc/aten/operators/op_method.h
# csrc/aten/operators/cnnl_ops.h
# csrc/aten/operators/cnnl_ops.cpp
# csrc/aten/operators/cnnl/cnnl_kernel.h

import os
import argparse
import json
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--use_bang', action='store_true',
                    help='select whether to generate the bang operator.')
args = parser.parse_args()

from code_template import CodeTemplate

try:
    from yaml import CLoader as YamlLoader
except ImportError:
    from yaml import Loader as YamlLoader

MLU_UNBOXEDONLY_WRAPPER_REGISTRATION = CodeTemplate("""\
.op(torch::RegisterOperators::options()
    .schema("${schema_string}")
    .impl_unboxedOnlyKernel<${return_type} (${formal_types}), &AtenMluType::${api_name}>(at::DispatchKey::MLU)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
""")

MLU_CUSTOM_WRAPPER_REGISTRATION = CodeTemplate("""\
.op("${schema_string}", torch::RegisterOperators::options()
    .aliasAnalysis(c10::AliasAnalysisKind::PURE_FUNCTION)
    .kernel<decltype(AtenMluCustomType::${api_name}),
    &AtenMluCustomType::${api_name}>(c10::DispatchKey::MLU))
""")

MLU_ATEN_TYPE_DECLARATION = CodeTemplate("""\
static ${return_type} ${api_name}(${type_formals_h});
""")

MLU_ATEN_CUSTOM_TYPE_DECLARATION = CodeTemplate("""\
static ${return_type} ${api_name}(${type_formals_h});
""")

MLU_ATEN_TYPE_DEFINITION = CodeTemplate("""\
${return_type} AtenMluType::${api_name}(${type_formals_c}) {
  return OP_DISPATCH(${op_name}, ${type_args});
}
""")

MLU_ATEN_CUSTOM_TYPE_DEFINITION = CodeTemplate("""\
${return_type} AtenMluCustomType::${api_name}(${type_formals_c}) {
  return OP_DISPATCH(${op_name}, ${type_args});
}
""")

OP_METHODS_DECLARATION = CodeTemplate("""\
virtual ${return_type} ${op_name}(${type_formals_h});
""")

CNNL_OPS_DECLARATION = CodeTemplate("""\
${return_type} ${op_name}(${type_formals_h}) override;
""")

CNNL_OPS_DEFINITION = CodeTemplate("""\
${return_type} CnnlOps::${op_name}(${type_formals_c}) {
  auto device = getTensorDevice({${tensor_list}});
  torch_mlu::mlu::MLUGuard guard(device);
  CNLOG(DBG)${logging_args};
  dump_layer_inputs("${op_name}", ${dump_args});
  CNNL_DISPATCH(${op_name}, ${cnnl_op_name}, ${type_args});
}
""")

BANG_OPS_DEFINITION = CodeTemplate("""\
${return_type} CnnlOps::${op_name}(${type_formals_c}) {
  auto device = getTensorDevice({${tensor_list}});
  torch_mlu::mlu::MLUGuard guard(device);
  CNLOG(DBG)${logging_args};
  dump_layer_inputs("${op_name}", ${dump_args});
  BANG_DISPATCH(${op_name}, ${bang_op_name}, ${type_args});
}
""")

CNNL_KERNEL_DECLARATION = CodeTemplate("""\
${return_type} ${cnnl_op_name}(${type_formals_h});
""")

BANG_KERNEL_DECLARATION = CodeTemplate("""\
${return_type} ${bang_op_name}(${type_formals_h});
""")

MLU_SUPPORTED_OPS = CodeTemplate("""\
"${supported_ops}",
""")

CNNL_OP_DEFINITION_EXC = {}

# register torchvision::nms using FROM_SHCEMA not PURE_FUNCTION
CUSTOM_WRAPPER_REGISTRATION_EXC = {'nms'}

VERSION_INFO = CodeTemplate("""\
std::string ${package_name} = "${version_num}";
""")

def version_collect(path):
    json_file = os.path.join(path, 'build.property')
    with open(json_file) as fp:
        json_dict = json.load(fp)
    versions = json_dict["build_requires"]
    versions["driver"] = json_dict["driver_requires"]
    version_collects = []
    version_catch = {}
    version_catch["package_name"] = "torch_mlu"
    version_catch["version_num"] = json_dict["version"]
    version_collects.append(version_catch)
    for key in versions:
        version_info = {}
        version_info["package_name"] = key + "_required_str"
        version_info["version_num"] = versions[key][1]
        version_collects.append(version_info)
    return version_collects


def select_declarations(path):
    # Parse mlu_functions.yaml
    with open(path, 'r') as f:
        declarations = yaml.load(f, Loader=YamlLoader)

    # Select essential declarations
    selected_declarations = []
    for declaration in declarations:
        tensor_list = []
        for arg in declaration['arguments']:
            if (arg['type'] == 'at::Tensor &') or (arg['type'] == 'const at::Tensor &'):
                tensor_list.append(arg['name'])
        declaration['tensor_list'] = ", ".join(tensor_list)
        declaration['type_formals_c'] = [arg['type'] + ' ' + arg['name'] \
                                        for arg in declaration['arguments']]
        declaration['type_formals_h'] = [arg['type'] + ' ' + arg['name'] + \
                       arg['default_value'] if 'default_value' in arg.keys() else arg['type'] + \
                       ' ' + arg['name'] for arg in declaration['arguments']]
        declaration['type_args'] = [arg['name'] for arg in declaration['arguments']]
        declaration['api_name'] = declaration['name']
        declaration['formal_types'] = [arg['type'] for arg in declaration['arguments']]
        declaration['op_name'] = declaration['api_name']
        declaration['logging_args'] = [''.join([' << ' + 'NAME(' + arg['name']+')' +
            ' << ' + 'TOSTR(' + arg['name'] + ')' for arg in declaration['arguments']])]
        declaration['dump_args'] = [','.join(['"' + arg['name'] + \
            '", ' + arg['name'] for arg in declaration['arguments']])]

        if declaration['derived_type'] == 'cnnl':
            declaration['cnnl_op_name'] = 'cnnl_' + declaration['op_name']

        if declaration['derived_type'] == 'bang':
            declaration['bang_op_name'] = 'bang_' + declaration['op_name']
        selected_declarations.append(declaration)

    return selected_declarations

def write(dirname, name, template, env):
    path = os.path.join(dirname, name)
    try:
        with open(path, 'r') as f:
            lines = f.read()
    except IOError:
        lines = None

    new_lines = template.substitute(env)
    if lines != new_lines:
        with open(path, 'w') as f:
            print("Writing {}".format(path))
            f.write(new_lines)
    else:
        print("Skipped writing {}".format(path))

def gen_files(out, aten_declarations, versions, template_path):
    aten_declarations = list(sorted(aten_declarations, key=lambda decl: decl['name']))

    ATEN_MLU_TYPE_DEFAULT_CPP = CodeTemplate.from_file(template_path + '/aten_mlu_type_default.cpp')
    ATEN_MLU_TYPE_DEFAULT_H = CodeTemplate.from_file(template_path + '/aten_mlu_type_default.h')
    ATEN_MLU_TYPE_H = CodeTemplate.from_file(template_path + '/aten_mlu_type.h')
    ATEN_MLU_TYPE_CPP = CodeTemplate.from_file(template_path + '/aten_mlu_type.cpp')
    ATEN_MLU_CUSTOM_TYPE_CPP = CodeTemplate.from_file(template_path + '/aten_mlu_custom_type.cpp')
    OP_METHODS_H = CodeTemplate.from_file(template_path + '/op_methods.h')
    CNNL_OPS_H = CodeTemplate.from_file(template_path + '/cnnl_ops.h')
    CNNL_OPS_CPP = CodeTemplate.from_file(template_path + '/cnnl_ops.cpp')
    CNNL_KERNEL_H = CodeTemplate.from_file(template_path + '/cnnl_kernel.h')
    BANG_KERNEL_H = CodeTemplate.from_file(template_path + '/bang_kernel.h')
    VERSION_CPP= CodeTemplate.from_file(template_path + '/version.cpp')


    unboxed_only_wrapper_registrations = []
    custom_wrapper_registrations = []
    type_declarations = []
    custom_type_declarations = []
    type_definitions = []
    custom_type_definitions = []
    op_methods_declarations = []
    cnnl_ops_declarations = []
    cnnl_ops_definitions = []
    cnnl_kernel_declarations = []
    bang_kernel_declarations = []
    version_info = []

    for info in versions:
        version_info.append(VERSION_INFO.substitute(info))

    for declaration in aten_declarations:
        if declaration['use_mlu_dispatcher'] == 'unboxed_only':
            unboxed_only_wrapper_registrations.append(
                MLU_UNBOXEDONLY_WRAPPER_REGISTRATION.substitute(declaration))
            type_declarations.append(MLU_ATEN_TYPE_DECLARATION.substitute(declaration))
            type_definitions.append(MLU_ATEN_TYPE_DEFINITION.substitute(declaration))
        else:
            assert declaration['use_mlu_dispatcher'] == 'custom'
            if declaration['op_name'] not in CUSTOM_WRAPPER_REGISTRATION_EXC:
                custom_wrapper_registrations.append(
                    MLU_CUSTOM_WRAPPER_REGISTRATION.substitute(declaration))
            custom_type_declarations.append(
                MLU_ATEN_CUSTOM_TYPE_DECLARATION.substitute(declaration))
            custom_type_definitions.append(
                MLU_ATEN_CUSTOM_TYPE_DEFINITION.substitute(declaration))

        op_methods_declarations.append(OP_METHODS_DECLARATION.substitute(declaration))

        if declaration['derived_type'] == 'cnnl':
            cnnl_ops_declarations.append(CNNL_OPS_DECLARATION.substitute(declaration))
            if declaration['op_name'] not in CNNL_OP_DEFINITION_EXC:
                cnnl_ops_definitions.append(CNNL_OPS_DEFINITION.substitute(declaration))
                cnnl_kernel_declarations.append(CNNL_KERNEL_DECLARATION.substitute(declaration))

        if args.use_bang and declaration['derived_type'] == 'bang':
            cnnl_ops_declarations.append(CNNL_OPS_DECLARATION.substitute(declaration))
            cnnl_ops_definitions.append(BANG_OPS_DEFINITION.substitute(declaration))
            bang_kernel_declarations.append(BANG_KERNEL_DECLARATION.substitute(declaration))

    env = {
        'unboxed_only_wrapper_registrations': unboxed_only_wrapper_registrations,
        'custom_wrapper_registrations': custom_wrapper_registrations,
        'type_declarations': type_declarations,
        'type_definitions': type_definitions,
        'custom_type_declarations': custom_type_declarations,
        'custom_type_definitions': custom_type_definitions,
        'op_methods_declarations': op_methods_declarations,
        'cnnl_ops_declarations': cnnl_ops_declarations,
        'cnnl_ops_definitions': cnnl_ops_definitions,
        'cnnl_kernel_declarations': cnnl_kernel_declarations,
        'bang_kernel_declarations': bang_kernel_declarations,
        'version_info': version_info,
    }

    generated_path = os.path.join(out, 'aten/generated')
    generated_autograd_path = os.path.join(generated_path, 'autograd')

    if not os.path.exists(generated_path):
        os.makedirs(generated_path)
    if not os.path.exists(generated_autograd_path):
        os.makedirs(generated_autograd_path)

    write(generated_path, 'aten_mlu_type_default.cpp', ATEN_MLU_TYPE_DEFAULT_CPP, env)
    write(generated_path, 'aten_mlu_type_default.h', ATEN_MLU_TYPE_DEFAULT_H, env)
    write(generated_path, 'aten_mlu_type.h', ATEN_MLU_TYPE_H, env)
    write(generated_path, 'aten_mlu_type.cpp', ATEN_MLU_TYPE_CPP, env)
    write(generated_path, 'aten_mlu_custom_type.cpp', ATEN_MLU_CUSTOM_TYPE_CPP, env)
    write(os.path.join(out, 'aten', 'operators'), 'op_methods.h', OP_METHODS_H, env)
    write(os.path.join(out, 'aten', 'operators'), 'cnnl_ops.h', CNNL_OPS_H, env)
    write(os.path.join(out, 'aten', 'operators'), 'cnnl_ops.cpp', CNNL_OPS_CPP, env)
    write(os.path.join(out, 'aten', 'operators', 'cnnl'), 'cnnl_kernel.h', CNNL_KERNEL_H, env)
    write(os.path.join(out, 'aten', 'operators', 'bang'), 'bang_kernel.h', BANG_KERNEL_H, env)
    write(os.path.join(out, 'aten', 'util'), 'version.cpp', VERSION_CPP, env)

if __name__ == '__main__':
    tool_path = os.path.dirname(os.path.abspath(__file__))
    torch_mlu_path = os.path.dirname(tool_path)
    mlu_declarations = select_declarations(os.path.join(tool_path, 'mlu_functions.yaml'))
    versions = version_collect(tool_path + "/../../script/release/")
    gen_files(os.path.join(
        torch_mlu_path, 'csrc'), mlu_declarations, versions,  os.path.join(tool_path, 'template'))
