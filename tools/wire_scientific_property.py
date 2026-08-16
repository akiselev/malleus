from pathlib import Path

p = Path('src/scientific_property.rs')
s = p.read_text()
s = s.replace('tangent: f64::from(derivative_input == Some(name.as_str())),', 'tangent: if derivative_input == Some(name.as_str()) { 1.0 } else { 0.0 },')
s = s.replace('Dual { value: f64::from(value), tangent: 0.0 }', 'Dual { value: if value { 1.0 } else { 0.0 }, tangent: 0.0 }')
p.write_text(s)

p = Path('src/lib.rs')
s = p.read_text()
if 'mod scientific_property;' not in s:
    s = s.replace('mod scientific;\n', 'mod scientific;\nmod scientific_property;\n', 1)
if 'pub use scientific_property::' not in s:
    anchor = '#[cfg(feature = "resolvent")]\npub use scientific::{CompiledKernelBundle, lower_kernel_bundle};\n'
    export = '''pub use scientific_property::{
    BranchKernel, CompareKernel, ExprKernel, GridAxisKernel, GridKernel, GuardKernel,
    PredicateKernel, ScientificPropertyError, ScientificPropertyKernel,
    ScientificPropertyModelKernel, ValidityPolicyKernel,
};
#[cfg(feature = "resolvent")]
pub use scientific_property::{ScientificPropertyLoweringError, lower_property_definition};
'''
    s = s.replace(anchor, anchor + export, 1)
p.write_text(s)
