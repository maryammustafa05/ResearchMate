import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  Sparkles, FileSearch, GitCompareArrows, Network,
  ArrowRight, Upload, MessageSquare, Quote, ChevronRight
} from 'lucide-react';
import './Landing.css';

function useReveal() {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { setVisible(true); obs.disconnect(); } },
      { threshold: 0.15 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return [ref, visible];
}

function RevealSection({ children, className = '' }) {
  const [ref, visible] = useReveal();
  return (
    <div ref={ref} className={`reveal ${visible ? 'reveal-visible' : ''} ${className}`}>
      {children}
    </div>
  );
}

export default function Landing() {
  return (
    <div className="landing">
      <Nav />
      <Hero />
      <ProofStrip />
      <HowItWorks />
      <FeatureGrid />
      <CTASection />
      <Footer />
    </div>
  );
}

function Nav() {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <nav className={`nav ${scrolled ? 'nav-scrolled' : ''}`}>
      <div className="nav-inner">
        <div className="nav-brand">
          <span className="nav-mark" aria-hidden="true">
            <Sparkles size={16} strokeWidth={2.25} />
          </span>
          ResearchMate
        </div>
        <div className="nav-links">
          <a href="#how">How it works</a>
          <a href="#features">Capabilities</a>
        </div>
        <Link to="/login" className="nav-cta">
          Open the assistant <ArrowRight size={15} />
        </Link>
      </div>
    </nav>
  );
}

function Hero() {
  return (
    <header className="hero">
      <div className="hero-glow" aria-hidden="true" />
      <div className="hero-grid" aria-hidden="true" />

      <div className="hero-content">
        <span className="hero-badge">
          <Sparkles size={13} strokeWidth={2} />
          Built for literature review, not lookup
        </span>

        <h1 className="hero-title">
          Ask your papers
          <span className="hero-title-glow"> anything.</span>
        </h1>

        <p className="hero-sub">
          Upload a paper, point at arXiv, or stack five at once — ResearchMate
          reads, retrieves, and cites the exact passage every answer came from.
          No more skimming 40 pages to find one sentence.
        </p>

        <div className="hero-actions">
          <Link to="/login" className="btn-primary">
            Start asking <ArrowRight size={16} />
          </Link>
          <a href="#how" className="btn-ghost">
            See how it works
          </a>
        </div>
      </div>

      <div className="hero-stage">
        <ChatPreview />
      </div>
    </header>
  );
}

function ChatPreview() {
  return (
    <div className="stage-card">
      <div className="stage-bar">
        <span className="stage-dot stage-dot-red" />
        <span className="stage-dot stage-dot-yellow" />
        <span className="stage-dot stage-dot-green" />
        <span className="stage-title">sign-language-recognition.pdf</span>
      </div>
      <div className="stage-body">
        <div className="bubble bubble-user">
          What accuracy did the CNN model achieve, and under what conditions?
        </div>
        <div className="bubble bubble-agent">
          <span className="bubble-label">
            <Sparkles size={11} /> ResearchMate
          </span>
          The model reached <strong>86.4% accuracy</strong> when trained on 500
          images, recognizing signs without a controlled background under
          low light.
          <div className="bubble-cite">
            <Quote size={12} />
            "Table-1 analysis — 500 images, 432 true results, 86.4% accuracy"
          </div>
        </div>
        <div className="bubble bubble-typing">
          <span className="dot" /><span className="dot" /><span className="dot" />
        </div>
      </div>
    </div>
  );
}

function ProofStrip() {
  const items = [
    'Grounded in the source, not memory',
    'Every claim cites its passage',
    'Compares any number of papers',
    'Pulls papers straight from arXiv',
  ];
  return (
    <div className="proof-strip">
      <div className="proof-track">
        {[...items, ...items].map((t, i) => (
          <span key={i} className="proof-item">{t}</span>
        ))}
      </div>
    </div>
  );
}

function HowItWorks() {
  const steps = [
    {
      icon: Upload,
      title: 'Bring your papers',
      body: 'Drop a PDF, paste an arXiv search, or queue up several at once. Each one is chunked and indexed in seconds.',
    },
    {
      icon: MessageSquare,
      title: 'Ask in plain language',
      body: 'No syntax to learn. Ask what you would ask a labmate — "what dataset did they use" or "how does this compare to the other paper."',
    },
    {
      icon: Quote,
      title: 'Get answers with receipts',
      body: 'Every response is grounded in retrieved passages, with the source excerpt shown next to the claim — never a confident guess.',
    },
  ];

  return (
    <section id="how" className="how">
      <RevealSection className="how-head">
        <span className="section-eyebrow">How it works</span>
        <h2 className="section-title">Three steps between you and a cited answer</h2>
      </RevealSection>

      <div className="how-steps">
        {steps.map((s, i) => (
          <RevealSection key={i} className="how-step">
            <div className="how-step-icon"><s.icon size={20} strokeWidth={1.75} /></div>
            <h3>{s.title}</h3>
            <p>{s.body}</p>
            {i < steps.length - 1 && <ChevronRight className="how-step-arrow" size={18} />}
          </RevealSection>
        ))}
      </div>
    </section>
  );
}

function FeatureGrid() {
  const features = [
    {
      icon: FileSearch,
      title: 'Single-paper Q&A',
      body: 'Upload one paper and interrogate it directly — methodology, results, limitations, all grounded in the actual text.',
    },
    {
      icon: Network,
      title: 'Search without leaving the chat',
      body: 'Give it a topic instead of a file. ResearchMate finds a relevant paper on arXiv, indexes it, and answers immediately.',
    },
    {
      icon: GitCompareArrows,
      title: 'Compare across papers',
      body: 'Stack two or ten papers and ask how their approaches differ. Each claim is labeled by which paper it came from.',
    },
  ];

  return (
    <section id="features" className="features">
      <RevealSection className="features-head">
        <span className="section-eyebrow">Capabilities</span>
        <h2 className="section-title">Built for the questions a literature review actually asks</h2>
      </RevealSection>

      <div className="features-grid">
        {features.map((f, i) => (
          <RevealSection key={i} className="feature-card">
            <div className="feature-icon"><f.icon size={22} strokeWidth={1.6} /></div>
            <h3>{f.title}</h3>
            <p>{f.body}</p>
          </RevealSection>
        ))}
      </div>
    </section>
  );
}

function CTASection() {
  return (
    <RevealSection className="cta">
      <div className="cta-glow" aria-hidden="true" />
      <h2>Stop re-reading the same paper for the third time today.</h2>
      <p>Upload one now — your first cited answer is a minute away.</p>
      <Link to="/login" className="btn-primary btn-large">
        Open ResearchMate <ArrowRight size={17} />
      </Link>
    </RevealSection>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-brand">
        <Sparkles size={14} /> ResearchMate
      </div>
      <p><p>Answers are grounded in the papers you provide, with sources shown for every claim.</p></p>
    </footer>
  );
}