# Hurricane Maria 30-day retrospective

This is one fixed **project reconstruction from official records**. It is not an official FEMA restoration percentage, an inverse fit, or a sensitivity study.

> The historical line is a project-derived index from official records. Policy lines are simulated alternatives in the frozen model, not observed or causal real-world outcomes.

| Series | Evidence type | Day 0 | Day 10 | Day 20 | Day 30 |
|---|---|---:|---:|---:|---:|
| Project reconstruction | official records + disclosed estimates | 10.9 | 34.2 | 42.7 | 51.3 |
| Shipped v4 | simulated alternative | 12.0 | 32.8 | 55.2 | 74.2 |
| Reactive heuristic | simulated alternative | 12.0 | 38.7 | 62.5 | 78.1 |

Values are derived-recovery-index points on a 0-100 display scale.

## Frozen scenario and replay

- Prepared-input contract SHA-256: `e52f0f5de499d73e078518c9a42a3a82e305dea67b4b2c18a46c294d7c56950d`
- Receipt SHA-256: `66eafd97e8336e2ad9e0a6fae1ba11dfe9cf3e03b0f1cc25a0888a24525329d4`
- Shipped ONNX SHA-256: `a9f5e9b41be57d7cd34623725a5ab4067aa75fbab16dc666cecc3c0a06c26483`
- Explicit no-secondary-shock tape SHA-256: `6940d42af85925d661f9856dd494d6c6c087650c93be35da957999a6d6f3685f`
- Daily budget: 180 abstract units; daily crew pool: 150 abstract units.
- Maria is encoded only in the post-landfall Day-0 service state.
- Both planners received the same explicit 30-day tape.
- Hard violations: 0 for both planners.
- Maximum conservation residual: exactly 0.0 for both planners.
- The canonical final split was not imported or evaluated.

## Service crosswalk

- Transport: official road/airport/port milestones; the Day-30 cross-mode value is a project estimate.
- Housing: qualitative official damage/recovery evidence converted to disclosed project-estimate anchors.
- Food and water: equal mean of potable-water and grocery availability.
- Healthcare: operational-hospital availability proxy.
- Public services: equal mean of restored electricity and operational cellular-site share.

The source manifest, raw observation table, selected/rejected alternatives, conversion formulas, and interpolation contract are tracked alongside this report.

## Official source manifest

| Agency | Title | Published | Retrieved | URL | Page/table locator | Archive filename | Bytes | SHA-256 / normalized-fact SHA-256 |
|---|---|---|---|---|---|---|---:|---|
| NOAA National Hurricane Center | Tropical Cyclone Report: Hurricane Maria (AL152017) | 2019-01-04 | 2026-08-13 | https://www.nhc.noaa.gov/data/tcr/AL152017_Maria.pdf | page 1, landfall chronology; pages 7-8, Puerto Rico impacts | nhc-maria-tcr.pdf | 7316824 | 9fcf9dbc2cf527fe318ab4b43d41cf775a6d28609253876829b64b43e15bfa75 |
| U.S. Department of Energy | Hurricanes Maria, Irma, and Harvey - September 22 Afternoon Event Summary (Report 43) | 2017-09-22 | 2026-08-13 | https://www.energy.gov/documents/hurricanes-maria-irma-and-harvey-event-summary-afternoon-september-22-2017pdf | page 1, Electricity Sector Summary; page 3, electricity outages table | doe-2017-09-22.pdf | 782881 | 6c4a01070306831e7e0e6a94cc408722ea9ec9d093ab79e0d01fbbdb7b56d34f |
| U.S. Department of Energy | Hurricanes Maria, Irma, and Harvey - October 2 Event Summary | 2017-10-02 | 2026-08-13 | https://www.energy.gov/documents/hurricanes-maria-irma-and-harvey-event-summary-october-2-2017pdf | page 2, Puerto Rico electricity restoration and transport notes | doe-2017-10-02.pdf | 269641 | 66a8bcc538ddf7c9a0c418f3d46b69ef049ee2c0b4111b19d75a9e5d520df579 |
| U.S. Department of Energy | Hurricanes Maria and Irma - October 20 Event Summary (Report 69) | 2017-10-20 | 2026-08-13 | https://www.energy.gov/documents/hurricanes-maria-and-irma-event-summary-october-20-2017pdf | page 1, Electricity Sector Summary; page 2, estimated outages and restored load | doe-2017-10-20.pdf | 192956 | fb4994b1b14e17186498c9154b97c0790a55aaa68d129bd208de328490b5ba7f |
| Federal Communications Commission | Communications Status Report for Areas Impacted by Hurricane Maria - September 22 | 2017-09-22 | 2026-08-13 | https://docs.fcc.gov/public/attachments/DOC-346855A1.pdf | page 2, Puerto Rico wireless-services summary | fcc-2017-09-22.pdf | 411198 | 58d022d7bd0abcbb81131482786af130ec97a2d89c5794ed76e182ce7ae476de |
| Federal Communications Commission | Communications Status Report for Areas Impacted by Hurricane Maria - October 5 | 2017-10-05 | 2026-08-13 | https://docs.fcc.gov/public/attachments/DOC-347091A1.pdf | page 2, Puerto Rico wireless-services summary | fcc-2017-10-05.pdf | 411486 | 1a830ac72c33e3cffb9bfa1d6efba5a751d62a0a00e81a51181bf84f8278721e |
| Federal Communications Commission | Communications Status Report for Areas Impacted by Hurricane Maria - October 18 | 2017-10-18 | 2026-08-13 | https://docs.fcc.gov/public/attachments/DOC-347308A1.pdf | pages 2-3, Puerto Rico wireless-services summary | fcc-2017-10-18.pdf | 502259 | bd0892f8e352358b16cf5c781745177283531010dceb7e174d18a93194947c3c |
| Federal Communications Commission | Communications Status Report for Areas Impacted by Hurricane Maria - October 19 | 2017-10-19 | 2026-08-13 | https://docs.fcc.gov/public/attachments/DOC-347339A2.pdf | pages 2-3, Puerto Rico wireless-services summary | fcc-2017-10-19.pdf | 488641 | 24028ef1db142c21938e26905231efc4e2eb74dce311307e66c1d2ab665b4927 |
| Federal Emergency Management Agency | Update on Federal Partners Supporting Survivors in Puerto Rico | 2017-09-30 | 2026-08-13 | https://www.fema.gov/fr/print/pdf/node/317292 | page 1, hospital, potable-water, grocery, and fuel restoration bullets | — | — | 0abaf5c90971e8446182ed7471457e30c7f49d093d4d26afb68da03b18fa7bc4 |
| Federal Emergency Management Agency | Hurricane Maria Update | 2017-11-06 | 2026-08-13 | https://www.fema.gov/vi/print/pdf/node/320687 | page 1, immediate and November communications, transport, power, water, grocery, and hospital milestones | — | — | 561470afec85601ae879608a8b0a57b0acee01c464e8352024bbd84283c1da20 |
| Federal Emergency Management Agency | Puerto Rico One Year after Hurricanes Irma and Maria | 2018-09-20 | 2026-08-13 | https://www.fema.gov/vi/print/pdf/node/335544 | page 2, Then and Now: exact 30-day power, water, road, and cellular milestones | — | — | 26645507c0c6a968aca6c19299e79d5c9856823b17a23c99468332fa73c008bf |
| U.S. Department of Defense / U.S. Army Corps of Engineers | Press briefing on support to hurricane relief efforts | 2017-09-22 | 2026-08-13 | https://www.defense.gov/News/Transcripts/Transcript/Article/1322049/department-of-defense-press-briefing-by-generals-rydholm-and-holland-in-the-pen/ | transcript, initial transport and infrastructure assessment | — | — | — |
| U.S. Department of Defense | DoD Accelerates Hurricane Relief, Response Efforts in Puerto Rico | 2017-09-30 | 2026-08-13 | https://www.defense.gov/News/News-Stories/Article/Article/1330501/dod-accelerates-hurricane-relief-response-efforts-in-puerto-rico/ | Puerto Rico situational update bullets | — | — | 8d98de043ae5c79e206b4b684ec352bcf9e5d2fc801498183bd12dbf6f70731d |
| U.S. Department of Defense / Federal Aviation Administration | Air National Guard Restores Air Traffic Control in Puerto Rico | 2017-10-02 | 2026-08-13 | https://www.defense.gov/News/News-Stories/Article/Article/1331249/air-national-guard-restores-air-traffic-control-in-puerto-rico/ | air-traffic restoration paragraph | — | — | 4eaa46a51379d9c2910bcad585270292347d7ebafff90a2b649342ef0b954342 |
| U.S. Department of Defense | Power Restoration Remains Top Concern in Puerto Rico, U.S. Virgin Islands | 2017-10-27 | 2026-08-13 | https://www.defense.gov/News/News-Stories/Article/Article/1356447/power-restoration-remains-top-concern-in-puerto-rico-us-virgin-islands/ | Puerto Rico recovery-status bullets | — | — | b0488faedd90197a1adc2231e61236f8d3671ca52dc6d98e447366e52cf00a9d |

Blocked FEMA/Defense.gov pages use the exact locator plus a tracked, reproducibly hashed normalized project fact record. These records are not quotations. Every downloaded byte object was verified before the input contract was frozen.

## Raw-statistic, conversion, and selection review

| ID | Date | Service / component | Sources | Reported statistic | Units | Denominator | Conversion | Final point | Decision | Selection reason |
|---|---|---|---|---|---|---|---|---:|---|---|
| transport_day0 | 2017-09-20 | transport / transport_access_proxy | fema_one_year_summary | 400 | miles of roads not shut down | 16700 | 400 / 16700 | 0.02395210 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| transport_day30_estimate | 2017-10-20 | transport / transport_access_proxy | fema_one_year_summary, dod_2017_10_02, fema_2017_11_06 | Road clearance has no 30-day island-wide denominator; normal air traffic by Day 12; all ports/airports by Day 47 | cross-mode transport access proxy | — | fixed Day-30 project estimate between the two dated official milestones | 0.45000000 | selected project estimate | Selected under the frozen denominator/coverage/date rules. |
| housing_day0_estimate | 2017-09-20 | housing / shelter_and_habitability_proxy | nhc_maria_tcr | Widespread significant building and home damage; no contemporaneous island-wide habitability percentage | qualitative damage assessment | — | fixed conservative project estimate; not an estimated percentage of habitable homes | 0.10000000 | selected project estimate | Selected under the frozen denominator/coverage/date rules. |
| housing_day30_estimate | 2017-10-20 | housing / shelter_and_habitability_proxy | nhc_maria_tcr, fema_2017_11_06 | Emergency roof and shelter work continued; no island-wide 30-day habitability series | qualitative recovery status | — | fixed modest-recovery project estimate; not an estimated percentage of habitable homes | 0.22000000 | selected project estimate | Selected under the frozen denominator/coverage/date rules. |
| food_water_day0_estimate | 2017-09-20 | food / potable_water_access | fema_one_year_summary | 20 | percent water service operational immediately after Maria | Puerto Rico water service | percent / 100 | 0.20000000 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| food_water_day7 | 2017-09-27 | food / potable_water_access | fema_2017_11_06 | 44 | percent of PRASA customers with running water | PRASA customers | percent / 100 | 0.44000000 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| food_water_day10 | 2017-09-30 | food / potable_water_access | fema_2017_09_30 | 45 | percent of customers with potable water | Puerto Rico water customers | percent / 100 | 0.45000000 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| food_water_day30_estimate | 2017-10-20 | food / potable_water_access | fema_one_year_summary | 69 | percent water service operational at 30 days | Puerto Rico water service | percent / 100 | 0.69000000 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| food_grocery_day0_estimate | 2017-09-20 | food / grocery_availability | fema_2017_09_30, nhc_maria_tcr | Island-wide baseline unavailable; 49% open on Day 10 | store availability | grocery and big-box stores | fixed pre-curve project estimate below the first observed Day-10 value | 0.10000000 | selected project estimate | Selected under the frozen denominator/coverage/date rules. |
| food_grocery_day10 | 2017-09-30 | food / grocery_availability | fema_2017_09_30 | 49 | percent open | grocery and big-box stores | percent / 100 | 0.49000000 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| food_grocery_day30_estimate | 2017-10-20 | food / grocery_availability | fema_2017_09_30, fema_2017_11_06 | 49% on Day 10 and more than 89% on Day 47 | linear interpolation to Day 30 using 89% | grocery stores | 0.49 + (0.89 - 0.49) * (20 / 37) | 0.70621622 | selected project estimate | Selected under the frozen denominator/coverage/date rules. |
| healthcare_day0_estimate | 2017-09-20 | healthcare / operational_hospital_availability | fema_2017_09_30, nhc_maria_tcr | No complete immediate island-wide assessment; 59 of 69 partially or fully operational by Day 10 | operational-capacity proxy | 69 hospitals | fixed conservative project estimate below the first complete assessment | 0.25000000 | selected project estimate | Selected under the frozen denominator/coverage/date rules. |
| healthcare_day10 | 2017-09-30 | healthcare / operational_hospital_availability | dod_2017_09_30, fema_2017_09_30 | 59 | partially or fully operational hospitals | 69 | 59 / 69 | 0.85507246 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| healthcare_day30_estimate | 2017-10-20 | healthcare / operational_hospital_availability | fema_2017_09_30, dod_2017_10_27 | 59/69 on Day 10 and 65/67 open on Day 37 | linear interpolation to Day 30 | reported hospital roster at each endpoint | (59/69) + ((65/67) - (59/69)) * (20 / 27) | 0.94031453 | selected project estimate | Selected under the frozen denominator/coverage/date rules. |
| public_power_day0 | 2017-09-20 | public_services / electricity | fema_2017_11_06, nhc_maria_tcr | 100 | percent without power | Puerto Rico island | 1 - outage_percent / 100 | 0.00000000 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| public_power_day2 | 2017-09-22 | public_services / electricity | doe_2017_09_22 | 100 | percent of customers without power | 1569796 | 1 - outage_percent / 100 | 0.00000000 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| public_power_day12 | 2017-10-02 | public_services / electricity | doe_2017_10_02 | 5.4 | percent of PREPA customers restored | PREPA customers | percent / 100 | 0.05400000 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| public_power_day30 | 2017-10-20 | public_services / electricity | fema_one_year_summary | 21 | percent of customers restored at 30 days | Puerto Rico electricity customers | percent / 100 | 0.21000000 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| public_cell_day0 | 2017-09-20 | public_services / cellular | fcc_2017_09_22 | 4.6% of reporting cell sites operational on Day 2 | two-day carry-back project estimate | reporting cell sites in Puerto Rico | carry Day-2 0.046 backward two days | 0.04600000 | selected project estimate | Selected under the frozen denominator/coverage/date rules. |
| rejected_fema_population_cell_day0 | 2017-09-20 | public_services / cellular | fema_2017_11_06 | 5 | percent of population with cellular service | Puerto Rico population | not used | 0.05000000 | rejected alternative | Population coverage is not commensurable with the selected FCC reporting-cell-site series. The reconstruction uses a disclosed carry-back from the first FCC site observation instead. |
| public_cell_day2 | 2017-09-22 | public_services / cellular | fcc_2017_09_22 | 95.4 | percent of cell sites out of service | reporting cell sites in Puerto Rico | 1 - outage_percent / 100 | 0.04600000 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| public_cell_day15 | 2017-10-05 | public_services / cellular | fcc_2017_10_05 | 84.6 | percent of cell sites out of service | reporting cell sites in Puerto Rico | 1 - outage_percent / 100 | 0.15400000 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| public_cell_day28 | 2017-10-18 | public_services / cellular | fcc_2017_10_18 | 71.2 | percent of cell sites out of service | 2,680 reporting cell sites | 1 - outage_percent / 100 | 0.28800000 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| public_cell_day29 | 2017-10-19 | public_services / cellular | fcc_2017_10_19 | 69.8 | percent of cell sites out of service | reporting cell sites in Puerto Rico | 1 - outage_percent / 100 | 0.30200000 | selected official observation | Selected under the frozen denominator/coverage/date rules. |
| public_cell_day30_carry | 2017-10-20 | public_services / cellular | fcc_2017_10_19 | 30.2% of reporting cell sites operational on Day 29 | one-day carry-forward project estimate | reporting cell sites in Puerto Rico | carry Day-29 0.302 forward one day | 0.30200000 | selected project estimate | Selected under the frozen denominator/coverage/date rules. |
| rejected_fema_cell_day30 | 2017-10-20 | public_services / cellular | fema_one_year_summary | 61 | percent of cellular sites operating at 30 days | Puerto Rico cellular sites | not used | 0.61000000 | rejected alternative | The retrospective 61% figure conflicts sharply with the contemporaneous FCC reporting-site series (30.2% operational on Day 29). The fixed curve preserves the internally consistent daily FCC metric and carries its nearest observation forward one day. |
| rejected_transport_priority_ports_day10 | 2017-09-30 | transport / transport_access_proxy | dod_2017_09_30 | 5 | priority seaports open or open with restrictions | 6 | not used | 0.83333333 | rejected alternative | The six priority seaports are a narrow, mode-specific roster and open-with-restrictions is not commensurable with the island-wide road-mile Day-0 denominator; using it as a transport percentage would overstate cross-mode access. |
| rejected_transport_airport_flow_day12 | 2017-10-02 | transport / transport_access_proxy | dod_2017_10_02 | normal hourly flight rate restored at Luis Munoz Marin International Airport | single-airport throughput milestone | normal hourly flight rate at one airport | not used | 1.00000000 | rejected alternative | A normal flow rate at one airport cannot be averaged as an island-wide roads/ports/airports restoration percentage; it is retained only as qualitative support for the Day-30 project estimate. |
| rejected_doe_peak_load_day30 | 2017-10-20 | public_services / electricity | doe_2017_10_20 | 18.5 | percent of normal peak load restored | 2,685 MW normal peak load | not used | 0.18500000 | rejected alternative | The selected FEMA 30-day customer-restoration figure has a customer denominator consistent with earlier series anchors; DOE peak load is a different measure. |
| rejected_fcc_population_coverage_day28 | 2017-10-18 | public_services / cellular | fcc_2017_10_18 | 61 | percent of population reportedly covered | Puerto Rico population | not used | 0.61000000 | rejected alternative | The fixed cellular series uses cell-site operational share consistently at all FCC dates; population coverage is a different measure. |

## Complete component interpolation, Day 0–30

Piecewise-linear interpolation is applied independently between selected anchors; the fixed carry-back/carry-forward values are explicitly marked as project estimates above.

| Day | Date | Transport proxy | Housing proxy | Water | Grocery | Hospitals | Power | Cell sites |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2017-09-20 | 0.02395210 | 0.10000000 | 0.20000000 | 0.10000000 | 0.25000000 | 0.00000000 | 0.04600000 |
| 1 | 2017-09-21 | 0.03815370 | 0.10400000 | 0.23428571 | 0.13900000 | 0.31050725 | 0.00000000 | 0.04600000 |
| 2 | 2017-09-22 | 0.05235529 | 0.10800000 | 0.26857143 | 0.17800000 | 0.37101449 | 0.00000000 | 0.04600000 |
| 3 | 2017-09-23 | 0.06655689 | 0.11200000 | 0.30285714 | 0.21700000 | 0.43152174 | 0.00540000 | 0.05430769 |
| 4 | 2017-09-24 | 0.08075849 | 0.11600000 | 0.33714286 | 0.25600000 | 0.49202898 | 0.01080000 | 0.06261538 |
| 5 | 2017-09-25 | 0.09496008 | 0.12000000 | 0.37142857 | 0.29500000 | 0.55253623 | 0.01620000 | 0.07092308 |
| 6 | 2017-09-26 | 0.10916168 | 0.12400000 | 0.40571429 | 0.33400000 | 0.61304348 | 0.02160000 | 0.07923077 |
| 7 | 2017-09-27 | 0.12336328 | 0.12800000 | 0.44000000 | 0.37300000 | 0.67355072 | 0.02700000 | 0.08753846 |
| 8 | 2017-09-28 | 0.13756487 | 0.13200000 | 0.44333333 | 0.41200000 | 0.73405797 | 0.03240000 | 0.09584615 |
| 9 | 2017-09-29 | 0.15176647 | 0.13600000 | 0.44666667 | 0.45100000 | 0.79456521 | 0.03780000 | 0.10415385 |
| 10 | 2017-09-30 | 0.16596807 | 0.14000000 | 0.45000000 | 0.49000000 | 0.85507246 | 0.04320000 | 0.11246154 |
| 11 | 2017-10-01 | 0.18016966 | 0.14400000 | 0.46200000 | 0.50081081 | 0.85933456 | 0.04860000 | 0.12076923 |
| 12 | 2017-10-02 | 0.19437126 | 0.14800000 | 0.47400000 | 0.51162162 | 0.86359667 | 0.05400000 | 0.12907692 |
| 13 | 2017-10-03 | 0.20857286 | 0.15200000 | 0.48600000 | 0.52243243 | 0.86785877 | 0.06266667 | 0.13738462 |
| 14 | 2017-10-04 | 0.22277445 | 0.15600000 | 0.49800000 | 0.53324324 | 0.87212087 | 0.07133333 | 0.14569231 |
| 15 | 2017-10-05 | 0.23697605 | 0.16000000 | 0.51000000 | 0.54405405 | 0.87638298 | 0.08000000 | 0.15400000 |
| 16 | 2017-10-06 | 0.25117765 | 0.16400000 | 0.52200000 | 0.55486487 | 0.88064508 | 0.08866667 | 0.16430769 |
| 17 | 2017-10-07 | 0.26537924 | 0.16800000 | 0.53400000 | 0.56567568 | 0.88490718 | 0.09733333 | 0.17461538 |
| 18 | 2017-10-08 | 0.27958084 | 0.17200000 | 0.54600000 | 0.57648649 | 0.88916929 | 0.10600000 | 0.18492308 |
| 19 | 2017-10-09 | 0.29378244 | 0.17600000 | 0.55800000 | 0.58729730 | 0.89343139 | 0.11466667 | 0.19523077 |
| 20 | 2017-10-10 | 0.30798403 | 0.18000000 | 0.57000000 | 0.59810811 | 0.89769349 | 0.12333333 | 0.20553846 |
| 21 | 2017-10-11 | 0.32218563 | 0.18400000 | 0.58200000 | 0.60891892 | 0.90195560 | 0.13200000 | 0.21584615 |
| 22 | 2017-10-12 | 0.33638723 | 0.18800000 | 0.59400000 | 0.61972973 | 0.90621770 | 0.14066667 | 0.22615385 |
| 23 | 2017-10-13 | 0.35058882 | 0.19200000 | 0.60600000 | 0.63054054 | 0.91047981 | 0.14933333 | 0.23646154 |
| 24 | 2017-10-14 | 0.36479042 | 0.19600000 | 0.61800000 | 0.64135135 | 0.91474191 | 0.15800000 | 0.24676923 |
| 25 | 2017-10-15 | 0.37899202 | 0.20000000 | 0.63000000 | 0.65216217 | 0.91900401 | 0.16666667 | 0.25707692 |
| 26 | 2017-10-16 | 0.39319361 | 0.20400000 | 0.64200000 | 0.66297298 | 0.92326612 | 0.17533333 | 0.26738462 |
| 27 | 2017-10-17 | 0.40739521 | 0.20800000 | 0.65400000 | 0.67378379 | 0.92752822 | 0.18400000 | 0.27769231 |
| 28 | 2017-10-18 | 0.42159681 | 0.21200000 | 0.66600000 | 0.68459460 | 0.93179032 | 0.19266667 | 0.28800000 |
| 29 | 2017-10-19 | 0.43579840 | 0.21600000 | 0.67800000 | 0.69540541 | 0.93605243 | 0.20133333 | 0.30200000 |
| 30 | 2017-10-20 | 0.45000000 | 0.22000000 | 0.69000000 | 0.70621622 | 0.94031453 | 0.21000000 | 0.30200000 |

## Complete final project reconstruction, Day 0–30

Each service is the disclosed component-weighted mean; Total is the equal arithmetic mean of all five services.

| Day | Date | Transport | Housing | Food and water | Healthcare | Public services | Total |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 2017-09-20 | 0.02395210 | 0.10000000 | 0.15000000 | 0.25000000 | 0.02300000 | 0.10939042 |
| 1 | 2017-09-21 | 0.03815370 | 0.10400000 | 0.18664285 | 0.31050725 | 0.02300000 | 0.13246076 |
| 2 | 2017-09-22 | 0.05235529 | 0.10800000 | 0.22328571 | 0.37101449 | 0.02300000 | 0.15553110 |
| 3 | 2017-09-23 | 0.06655689 | 0.11200000 | 0.25992857 | 0.43152174 | 0.02985385 | 0.17997221 |
| 4 | 2017-09-24 | 0.08075849 | 0.11600000 | 0.29657143 | 0.49202898 | 0.03670769 | 0.20441332 |
| 5 | 2017-09-25 | 0.09496008 | 0.12000000 | 0.33321428 | 0.55253623 | 0.04356154 | 0.22885443 |
| 6 | 2017-09-26 | 0.10916168 | 0.12400000 | 0.36985714 | 0.61304348 | 0.05041539 | 0.25329554 |
| 7 | 2017-09-27 | 0.12336328 | 0.12800000 | 0.40650000 | 0.67355072 | 0.05726923 | 0.27773665 |
| 8 | 2017-09-28 | 0.13756487 | 0.13200000 | 0.42766667 | 0.73405797 | 0.06412308 | 0.29908252 |
| 9 | 2017-09-29 | 0.15176647 | 0.13600000 | 0.44883333 | 0.79456521 | 0.07097692 | 0.32042839 |
| 10 | 2017-09-30 | 0.16596807 | 0.14000000 | 0.47000000 | 0.85507246 | 0.07783077 | 0.34177426 |
| 11 | 2017-10-01 | 0.18016966 | 0.14400000 | 0.48140541 | 0.85933456 | 0.08468462 | 0.34991885 |
| 12 | 2017-10-02 | 0.19437126 | 0.14800000 | 0.49281081 | 0.86359667 | 0.09153846 | 0.35806344 |
| 13 | 2017-10-03 | 0.20857286 | 0.15200000 | 0.50421622 | 0.86785877 | 0.10002564 | 0.36653470 |
| 14 | 2017-10-04 | 0.22277445 | 0.15600000 | 0.51562162 | 0.87212087 | 0.10851282 | 0.37500595 |
| 15 | 2017-10-05 | 0.23697605 | 0.16000000 | 0.52702702 | 0.87638298 | 0.11700000 | 0.38347721 |
| 16 | 2017-10-06 | 0.25117765 | 0.16400000 | 0.53843244 | 0.88064508 | 0.12648718 | 0.39214847 |
| 17 | 2017-10-07 | 0.26537924 | 0.16800000 | 0.54983784 | 0.88490718 | 0.13597435 | 0.40081972 |
| 18 | 2017-10-08 | 0.27958084 | 0.17200000 | 0.56124325 | 0.88916929 | 0.14546154 | 0.40949098 |
| 19 | 2017-10-09 | 0.29378244 | 0.17600000 | 0.57264865 | 0.89343139 | 0.15494872 | 0.41816224 |
| 20 | 2017-10-10 | 0.30798403 | 0.18000000 | 0.58405405 | 0.89769349 | 0.16443589 | 0.42683349 |
| 21 | 2017-10-11 | 0.32218563 | 0.18400000 | 0.59545946 | 0.90195560 | 0.17392307 | 0.43550475 |
| 22 | 2017-10-12 | 0.33638723 | 0.18800000 | 0.60686486 | 0.90621770 | 0.18341026 | 0.44417601 |
| 23 | 2017-10-13 | 0.35058882 | 0.19200000 | 0.61827027 | 0.91047981 | 0.19289744 | 0.45284727 |
| 24 | 2017-10-14 | 0.36479042 | 0.19600000 | 0.62967567 | 0.91474191 | 0.20238461 | 0.46151852 |
| 25 | 2017-10-15 | 0.37899202 | 0.20000000 | 0.64108108 | 0.91900401 | 0.21187179 | 0.47018978 |
| 26 | 2017-10-16 | 0.39319361 | 0.20400000 | 0.65248649 | 0.92326612 | 0.22135898 | 0.47886104 |
| 27 | 2017-10-17 | 0.40739521 | 0.20800000 | 0.66389190 | 0.92752822 | 0.23084615 | 0.48753230 |
| 28 | 2017-10-18 | 0.42159681 | 0.21200000 | 0.67529730 | 0.93179032 | 0.24033333 | 0.49620355 |
| 29 | 2017-10-19 | 0.43579840 | 0.21600000 | 0.68670271 | 0.93605243 | 0.25166667 | 0.50524404 |
| 30 | 2017-10-20 | 0.45000000 | 0.22000000 | 0.69810811 | 0.94031453 | 0.25600000 | 0.51288453 |

## Initial-condition clipping review

| Service | Reconstructed Day 0 | Scenario Day 0 | Clipped |
|---|---:|---:|---|
| transport | 0.02395210 | 0.05000000 | true |
| housing | 0.10000000 | 0.10000000 | false |
| food | 0.15000000 | 0.15000000 | false |
| healthcare | 0.25000000 | 0.25000000 | false |
| public_services | 0.02300000 | 0.05000000 | true |

## Synthetic benchmark results (separate evidence scope)

These rows are derived from the retained canonical aggregate evidence bound in the frozen input contract; this retrospective did not import or evaluate the final roster.

| Method | Solved | Rate | Classification | Detail |
|---|---:|---:|---|---|
| Clairvoyant oracle | 182/200 | 0.910 | Privileged; not a submission baseline | Complete future-tape knowledge; anytime achieved lower bound. |
| v4 PPO (shipped) | 163/200 | 0.815 | Shipped policy | Single owner-authorized final evaluation. |
| Tuned constant rule | 147/200 | 0.735 | Hand-coded planner | Strongest hand-coded comparator. |
| Preparedness teacher | 139/200 | 0.695 | Public deterministic regression | Original behavior-cloning teacher. |
| Selected MPC | 135/200 | 0.675 | Causal diagnostic | Selected receding-horizon planner, k=5. |
| Legacy shipped policy | 125/200 | 0.625 | Retired regression fixture | Legacy ONNX comparator. |
| Reactive heuristic | 72/200 | 0.360 | Public deterministic regression | Simple reactive allocation heuristic. |
