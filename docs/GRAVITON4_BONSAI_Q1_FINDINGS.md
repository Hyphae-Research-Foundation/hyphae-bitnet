# Graviton4 Bonsai Q1 Findings

A temporary AWS `r8g.metal-24xl` was used to establish the first Bonsai 27B
`Q1_0` baseline on Graviton4. The host exposed 96 physical Neoverse V2 cores,
768 GiB RAM, SVE/SVE2, DOTPROD, and I8MM.

The current Celiums engine built successfully after fixing an ARM-only scope
error in the inherited I2_S fallback. Q1_0 used the ordinary ARM path; the
AVX-512 repack is x86-specific and was not selected.

| Threads | pp128 tok/s | tg128 tok/s |
| ---: | ---: | ---: |
| 1 | 0.53 | 0.50 |
| 8 | 4.19 | 3.75 |
| 24 | 11.66 | 9.13 |
| 48 | 20.98 | 14.27 |
| 72 | 29.62 | 17.58 |
| 96 | 37.82 | 17.92 |

Graviton4 matched the optimized Xeon prefill result and exceeded its peak
decode throughput by approximately 63%, despite lacking a Q1-specific ARM
repack in the current Celiums tree. Scaling remained positive through 96
physical cores, unlike the DigitalOcean guest where 60 exposed CPUs likely
included SMT siblings.

Peak RSS was only 3.82 GiB because Q1 weights stayed mmap-backed rather than
being duplicated into a repacked buffer. The next ARM step is a four- or
eight-row Q1 repack using DOTPROD/I8MM, followed by an SVE2-native wider tile.

The baseline archive SHA256 is
`bbf2fdb6e7208dfd7d5fb7871d2489243d8f974e89932e617b6702b94f31ce84`.
The EC2 host, EBS volume, VPC, subnet, route table, internet gateway, security
group, and temporary key pair were removed after the archive was downloaded.
