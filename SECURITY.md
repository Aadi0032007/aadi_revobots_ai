<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0a0a0f,40:1a0a0a,80:3c0d0d,100:5c1010&height=160&section=header&text=Security%20Policy&fontSize=40&fontColor=ffffff&fontAlignY=42&desc=aadi_revobots_ai%20×%20REVOBOTS&descAlignY=62&descSize=16&animation=fadeIn" alt="banner"/>

</div>

---

## ⚠️ Important Context

**aadi_revobots_ai** is the AI control and training stack for [Revobots](https://revobots.ai) robotic systems. Security vulnerabilities in this codebase can have **real-world physical consequences** — including unsafe robot behaviour, unintended actuation, or harm to people and equipment in industrial environments.

We take all security reports seriously and ask that you do the same when disclosing them.

---

## Supported Versions

| Version | Supported |
|---|---|
| `main` (latest) | ✅ Actively maintained |
| Tagged releases | ✅ Critical patches backported where feasible |
| Older branches | ❌ No active security support |

Always run the latest commit on `main` for the most up-to-date security patches.

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report all security concerns privately and directly to the project maintainer:

**Aditya Raj** — Project Owner  
📧 **[ms.adityaraj@gmail.com](mailto:ms.adityaraj@gmail.com)**  
🌐 **[revobots.ai](https://revobots.ai)**

---

### What to Include in Your Report

Please provide as much of the following as possible:

- **Description** — a clear summary of the vulnerability
- **Affected component** — which robot, policy, dataset pipeline, or API is affected
- **Reproduction steps** — minimal steps to trigger the issue
- **Impact assessment** — what an attacker or faulty state could cause (e.g. uncontrolled motor actuation, data exfiltration, policy poisoning)
- **Suggested fix** — if you have one
- **Your contact details** — so we can follow up

---

## Our Response Commitment

| Milestone | Target Timeframe |
|---|---|
| Acknowledgement of your report | Within **48 hours** |
| Initial severity assessment | Within **5 business days** |
| Fix or mitigation for critical issues | Within **14 days** |
| Public disclosure (coordinated) | After fix is deployed |

We will keep you informed at each stage and coordinate the disclosure timeline with you.

---

## Severity Classification

We assess vulnerabilities using the following categories:

| Severity | Description | Examples |
|---|---|---|
| 🔴 **Critical** | Can cause physical harm or unsafe robot behaviour | Bypassing joint limits, disabling emergency stop, arbitrary motor commands |
| 🟠 **High** | Significant system compromise or data breach | Remote code execution, credential theft, policy model poisoning |
| 🟡 **Medium** | Partial impact with limited scope | Denial of service, incorrect sensor data, dataset tampering |
| 🟢 **Low** | Minimal real-world impact | Minor info disclosure, non-exploitable edge cases |

---

## Safety-Critical Code Paths

The following areas are treated with the highest priority due to their potential for physical harm:

- **Joint velocity and position limits** — `src/revobots/robots/*/`
- **Emergency stop and fault handling** — any `stop()`, `disconnect()`, or safety interlock logic
- **Teleoperation input validation** — `src/revobots/teleop/`
- **Policy inference pipeline** — unchecked model outputs sent directly to actuators
- **Dataset integrity** — poisoned training data leading to unsafe learned behaviour

Any report touching these areas is automatically escalated to **Critical** pending investigation.

---

## Responsible Disclosure Policy

We follow coordinated disclosure:

1. You report the vulnerability privately to us
2. We investigate, develop a fix, and test it
3. We release the fix and notify affected users
4. We publicly acknowledge your contribution (with your permission)

We ask that you:

- Give us a reasonable time to respond before any public disclosure
- Do not exploit the vulnerability beyond what is necessary to demonstrate it
- Do not access, modify, or exfiltrate data belonging to Revobots customers
- Do not disrupt production robotic systems or customer deployments

---

## Acknowledgements

We gratefully acknowledge responsible security researchers who help keep this project and its users safe. With your permission, your name will be listed here following coordinated disclosure.

---

## Out of Scope

The following are generally **not** considered security vulnerabilities for this project:

- Bugs that require physical access to the robot hardware to exploit
- Issues in third-party dependencies (report those upstream — e.g. to LeRobot or HuggingFace)
- Performance degradation without a security impact
- Theoretical issues with no practical exploit path

---

<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:5c1010,50:3c0d0d,100:0a0a0f&height=100&section=footer&animation=fadeIn" alt="footer"/>

**© 2025 Revobots · [revobots.ai](https://revobots.ai)**  
*Physical AI demands physical responsibility.*

</div>