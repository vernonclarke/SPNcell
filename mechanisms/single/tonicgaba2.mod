TITLE gaba leak 

COMMENT
If want ohmic gaba use tonicgaba1
From Pavlov et al., 2009
Outwardly Rectifying Tonically Active GABAA Receptors in Pyramidal Cells Modulate Neuronal Offset, Not Gain 
ENDCOMMENT

NEURON{
    NONSPECIFIC_CURRENT i

UNITS {
	(mA)  = (milliamp)
	(mV)  =  (millivolt)
}

PARAMETER {

ASSIGNED{

BREAKPOINT {

FUNCTION a(v(mV)) {

FUNCTION b(v(mV)) {