PRO irfm,v,k

; computes the radius from the infrared flux following Blackwell & Lynas-Gray 1994
; (A&A 282, 899, 1994)

; input: v and k magnitudes (corrected for extinction, if applicable)
; output: angular diameter

; NOTE: seems to work only up to late K spectral type;
; Teff smaller than about 4000K as applicable for M stars cannot be calculated

; constants:
pi=3.1415926535897932d0
zweipi=2.d0*pi
sigma = 5.67032d-08  ; Stefan Boltzmann constant in [Watt/m^2/K^4]

vmk = v-k

; non binary parameters
a = 8906.d0
b = -2625.d0
c = 363.2d0

; binary parameters
;a =  8825.d0
;b = -2548.d0
;c = 343.9d0

; mix
;a = 8862.d0
;b = -2583.d0
;c = 353.1d0

; effective temperature teff
teff = a+b*vmk+c*vmk^2.d0

; reduced flux phi
phi = (2.47930d0-0.04810d0*vmk-0.01475d0*vmk^2.d0+0.09233d0*vmk^3.d0)*1.d-08

; integrated flux
flux = 10.d0^(-0.4d0*v)*phi

; angular diameter theta
theta = sqrt(4.d0*flux/(sigma*teff^4.d0))
theta = theta/zweipi*360.d0*3600.d0*1000.d0  ; convert from radians to mas

print,'angular diameter: ',theta,' mas'
print,'effective temperature: ',teff,' K'

stop
END 
