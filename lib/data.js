import { FaGithub, FaLinkedin} from "react-icons/fa";
import { IoIosMail } from "react-icons/io";
import Image from "next/image";


export const socials = [
  {
    title: "Github",
    url: "https://github.com/Vincent-Maladiere",
    icon: <FaGithub />
  },
  {
    title: "Linkedin",
    url: "https://linkedin.com/in/vincent-maladiere",
    icon: <FaLinkedin />
  },
  {
    title: "mail",
    url: "mailto:maladiere.vincent@gmail.com",
    icon: <IoIosMail />

  },
  {
    title: ":probabl. team",
    url: "https://probabl.ai/about",
    icon: <Image alt=":probabl. team" src="/probabl-fav.png" height={16} width={16}/>
  },
  {
    title: "Inria's scikit-learn team",
    url: "https://team.inria.fr/soda/team-members/",
    icon: <Image alt="scikit-learn team" src="/sk-fav.png" height={16} width={16}/>
  },
];

export const navLinks = [
  {
    title: "Proba ML",
    url: "/proba-ml/home",
  },
  {
    title: "About",
    url: "/about",
  },
  {
    title: "Contact",
    url: "mailto:maladiere.vincent@gmail.com",
  },
];

export const libs = [
  {
    title: "scikit-learn (see my contributions)",
    url: "https://github.com/scikit-learn/scikit-learn/pulls?q=is:pr+sort:updated-desc+author:Vincent-Maladiere+",
    img: "/scikit.png",
  },
  {
    title: "skrub",
    url: "https://github.com/skrub-data/skrub",
    img: "/skrub.png",
  },
  {
    title: "hazardous",
    url: "https://github.com/soda-inria/hazardous",
    img: "/hazardous.png",
  },
  {
    title: "skfolio",
    url: "https://github.com/skfolio/skfolio",
    img: "/skfolio.svg",
  }
];

export const guides = [
  {
    title: "Proba ML",
    url: "/proba-ml/home",
    img: "/proba-ml.png",
    desc: "An all-in-one, simplified mathematical guide of the most popular ML models, based on Kevin Murphy's book.",
  },
  {
    title: "apply(conf)",
    url: "/apply-2022/home",
    img: "/apply-conf.png",
    desc: "How to create an ML Platform? A digest of all the talks and wisdom given at apply(conf) 2022.",
  },
  {
    title: "Contributing",
    url: "/contributing-scikit-learn/home",
    img: "/contributing.png",
    desc: "A beginner-friendly tutorial to start contributing to any open source project.",
  },
];

export const courses = [
  {
    title: "Python for Data Science",
    school: "Polytechnique",
    img: "/xhec.png",
    url: "/xhec",
    desc: "Lectures on the main linear algebra and Python concepts for M2 students.",
  },
  {
    title: "Survival analysis",
    school: "Dauphine",
    img: "/dauphine.png",
    url: "https://vincent-maladiere.github.io/survival-analysis-demo/",
    desc: "An introduction to survival analysis and hazardous.",
  },
];

export const conferences = [
  {
    title: "skrub: Bringing everything into the model",
    venue: "EuroScipy 2024",
    city: "Szczecin",
    img: "/euroscipy.png",
    url: "/euroscipy-2024",
  },
  {
    title: "Introducing hazardous",
    venue: "EuroScipy 2023",
    city: "Basel",
    img: "/euroscipy.png",
    url: "/euroscipy-2023",
  },
  {
    title: "Predictive survival analysis and competing risks modelling",
    venue: "JupyterCon 2023",
    city: "Paris",
    img: "/jupytercon.jpeg",
    url: "/jupytercon-2023",
  },
  {
    title: "skrub: Prepping tables for machine learning",
    venue: "Axa Summit 2022",
    city: "Köln",
    img: "/axa.png",
    url: "/axa-2022",
  },
  {
    title: "Survival Analysis lightning talk",
    venue: "Inria 2022",
    city: "Palaiseau",
    img: "/inria.png",
    url: "/inria-2022",
  },
  {
    title: "Linear Algebra and Optimization Demo",
    venue: "EuroScipy 2022",
    city: "Basel",
    img: "/euroscipy.png",
    url: "/euroscipy-2022",
  },
];

export const notebooks = [
  {
    title: "Understanding the Amazon from space",
    model: "tSNE, Resnet",
    url: "https://www.kaggle.com/code/vincentmaladiere/pytorch-resnet18-0-93",
    img: "/amazon.png",
  },
  {
    title: "Audio detection of chainsaw",
    model: "FFT, Resnet",
    url: "https://www.kaggle.com/code/vincentmaladiere/avalanche-chainsaw-audio-training",
    img: "/audio-detection.png",
  },
  {
    title: "Zenly geo clustering challenge",
    model: "DBSCAN, HDBSCAN",
    url: "https://www.kaggle.com/code/vincentmaladiere/zenly-challenge-density-clustering-of-gps-data",
    img: "/zenly.png",
  },
  {
    title: "Agents lab, workflow and agentic systems",
    model: "pydantic-ai, OpenAI, Groq",
    url: "https://vincent-maladiere.github.io/agents-lab/",
    img: "/agent-crew-cropped.png"
  }
];
