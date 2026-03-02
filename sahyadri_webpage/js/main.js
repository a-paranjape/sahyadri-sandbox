/* ═══════════════════════════════════════════
   Sahyadri — Shared JavaScript
   ═══════════════════════════════════════════ */

(function () {
  "use strict";

  /* ─── Determine current page from filename ─── */
  var path = window.location.pathname;
  var file = path.substring(path.lastIndexOf("/") + 1) || "index.html";
  var PAGE_ID = file.replace(".html", "");
  if (PAGE_ID === "index") PAGE_ID = "home";

  /* ─── Page definitions ─── */
  var PAGES = [
    { id: "home",           href: "index.html",          label: "Home" },
    { id: "overview",       href: "overview.html",       label: "Overview" },
    { id: "technical",      href: "technical.html",      label: "Technical" },
    { id: "data",           href: "data.html",           label: "Data Products" },
    { id: "science",        href: "science.html",        label: "Science" },
    { id: "learn",          href: "learn.html",          label: "Learning Hub" },
    { id: "visualizations", href: "visualizations.html", label: "Visualizations" },
    { id: "publications",   href: "publications.html",   label: "Publications" },
  ];

  /* ─── Build & inject navigation ─── */
  function buildNav() {
    var target = document.getElementById("site-nav");
    if (!target) return;

    var desktopLinks = PAGES.filter(function (p) { return p.id !== "home"; })
      .map(function (p) {
        var cls = p.id === PAGE_ID ? ' class="active"' : "";
        return '<a href="' + p.href + '"' + cls + ">" + p.label + "</a>";
      })
      .join("");

    var mobileLinks = PAGES.map(function (p) {
      var cls = p.id === PAGE_ID ? ' class="active"' : "";
      return '<a href="' + p.href + '"' + cls + ">" + p.label + "</a>";
    }).join("");

    target.innerHTML =
      '<nav class="nav" id="navbar">' +
        '<div class="nav-inner">' +
          '<a class="nav-logo" href="index.html"><span>S</span>ahyadri</a>' +
          '<div class="nav-links">' + desktopLinks + "</div>" +
          '<button class="nav-hamburger" id="hamburger" aria-label="Menu">☰</button>' +
        "</div>" +
      "</nav>" +
      '<div class="mobile-menu" id="mobileMenu">' + mobileLinks + "</div>";

    /* Hamburger toggle */
    var btn = document.getElementById("hamburger");
    var mm = document.getElementById("mobileMenu");
    if (btn && mm) {
      btn.addEventListener("click", function () {
        var open = mm.classList.toggle("open");
        btn.textContent = open ? "✕" : "☰";
      });
    }

    /* Scroll shadow */
    window.addEventListener("scroll", function () {
      var nb = document.getElementById("navbar");
      if (nb) nb.classList.toggle("scrolled", window.scrollY > 20);
    });
  }

  /* ─── Build & inject footer ─── */
  function buildFooter() {
    var target = document.getElementById("site-footer");
    if (!target) return;

    target.innerHTML =
      '<footer class="footer">' +
        '<div class="footer-name">Sahyadri</div>' +
        '<div class="footer-desc">' +
          "High-resolution N-body simulations for precision low-redshift cosmology. " +
          "Designed for DESI BGS, 4MOST, and next-generation survey science." +
        "</div>" +
        '<div class="footer-copy">© 2025 Sahyadri Collaboration</div>' +
      "</footer>";
  }

  /* ─── Accordion (Learning Hub page) ─── */
  function initAccordion() {
    var container = document.getElementById("accordionContainer");
    if (!container) return;

    var topics = [
      {
        title: "What are N-body Simulations?",
        body: [
          "N-body simulations are computational methods that follow the gravitational evolution of a large number of particles representing dark matter in the Universe. Starting from initial conditions set by early-Universe physics (derived from the Cosmic Microwave Background), these simulations evolve particles forward in time under Newtonian gravity to produce a detailed picture of how cosmic structures — filaments, walls, voids, and halos — form and evolve.",
          "GADGET-4, the code used for Sahyadri, employs a Tree-PM algorithm: short-range forces are computed via a hierarchical tree, while long-range forces use a particle-mesh Fourier method. This hybrid approach balances accuracy and performance for cosmological volumes.",
        ],
      },
      {
        title: "Cosmological Parameters Explained",
        body: [
          "The standard cosmological model (ΛCDM) is described by a handful of parameters. Ωₘ (Omega matter) is the fraction of the Universe's energy density in matter — higher Ωₘ means more clustering. The Hubble parameter h relates to the expansion rate as H₀ = 100h km/s/Mpc and affects distances and volumes.",
          "nₛ (scalar spectral index) describes how density fluctuations vary with scale; nₛ ≈ 1 means nearly scale-invariant. Aₛ (scalar amplitude) sets the overall amplitude of initial fluctuations — larger Aₛ leads to more structure. Ωᵦ (baryon density) affects the BAO scale and galaxy formation. Ωₖ (curvature) measures spatial curvature; Ωₖ = 0 means a flat Universe, consistent with current observations.",
        ],
      },
      {
        title: "Dark Matter Halos and Halo Finding",
        body: [
          "Dark matter halos are gravitationally bound concentrations of dark matter that host galaxies. They form when overdense regions in the initial density field collapse and virialize. The mass, concentration, spin, and environment of a halo strongly influence the properties of the galaxy it hosts.",
          "Rockstar, the halo finder used in Sahyadri, operates in 6D phase space (positions + velocities). This makes it more robust than position-only finders for identifying subhalos within larger hosts — critical for studying satellite galaxies.",
        ],
      },
      {
        title: "The Halo Occupation Distribution (HOD)",
        body: [
          "HOD modeling is a statistical framework connecting dark matter halos to galaxies. The core idea: the number and properties of galaxies in a halo depend primarily on halo mass. The HOD specifies P(N|M), the probability of finding N galaxies of a given type in a halo of mass M.",
          "A typical HOD has two components: central galaxies (one per halo above a threshold mass) and satellite galaxies (following a power law in halo mass). By calibrating HOD parameters against observed clustering, one can populate simulated halo catalogs with realistic galaxy distributions — a key use case for Sahyadri.",
        ],
      },
      {
        title: "Matter Power Spectrum P(k)",
        body: [
          "The matter power spectrum P(k) quantifies the amplitude of density fluctuations as a function of spatial scale k (in h/Mpc). Large k corresponds to small scales. P(k) encodes the statistical properties of the cosmic density field and is one of the most fundamental measurements in cosmology.",
          "At large scales (small k), P(k) retains the shape from initial conditions and linear growth. At smaller scales (large k > 0.1 h/Mpc), non-linear gravitational collapse significantly modifies P(k). Sahyadri's resolution is crucial for accurately capturing this non-linear regime where most cosmological information resides.",
        ],
      },
      {
        title: "Voronoi Volume Functions (VVF)",
        body: [
          "The Voronoi Volume Function is a beyond-two-point statistic that characterizes the local environment of objects. For each object, a Voronoi tessellation assigns a polyhedral cell whose volume is inversely related to the local density.",
          "The VVF — the distribution of these Voronoi volumes — captures information about the cosmic web that traditional two-point statistics miss. Sahyadri has demonstrated that the VVF is sensitive to cosmological parameters, making it a promising observable for next-generation surveys.",
        ],
      },
      {
        title: "Fisher Matrix Forecasts",
        body: [
          "Fisher matrix analysis is a method for forecasting how well future experiments can constrain model parameters. It uses derivatives of observables with respect to parameters to estimate expected uncertainties.",
          "For Sahyadri, seed-matched simulations with different cosmological parameters enable clean numerical computation of these derivatives. The Fisher information matrix then gives the expected parameter covariance, essential for understanding which statistics and survey configurations best constrain cosmology.",
        ],
      },
      {
        title: "DESI BGS and 4MOST Surveys",
        body: [
          "DESI (Dark Energy Spectroscopic Instrument) is a next-generation spectroscopic survey mapping tens of millions of galaxies. The Bright Galaxy Survey (BGS) targets low-redshift (z < 0.4) galaxies, providing the densest spectroscopic sample of the nearby Universe.",
          "4MOST (4-metre Multi-Object Spectroscopic Telescope) will conduct complementary surveys from the Southern Hemisphere. Together, these surveys will probe structure formation, galaxy evolution, and cosmology at unprecedented precision — exactly the regime Sahyadri is optimized for.",
        ],
      },
    ];

    topics.forEach(function (t) {
      var item = document.createElement("div");
      item.className = "accordion-item";

      var btn = document.createElement("button");
      btn.className = "accordion-btn";
      btn.innerHTML =
        "<span>" + t.title + '</span><span class="accordion-icon">+</span>';

      var body = document.createElement("div");
      body.className = "accordion-body";
      t.body.forEach(function (p) {
        var el = document.createElement("p");
        el.textContent = p;
        body.appendChild(el);
      });

      btn.addEventListener("click", function () {
        var isOpen = btn.classList.contains("open");
        container.querySelectorAll(".accordion-btn").forEach(function (b) {
          b.classList.remove("open");
        });
        container.querySelectorAll(".accordion-body").forEach(function (b) {
          b.classList.remove("open");
        });
        if (!isOpen) {
          btn.classList.add("open");
          body.classList.add("open");
        }
      });

      item.appendChild(btn);
      item.appendChild(body);
      container.appendChild(item);
    });
  }

  /* ─── Image Slider (Visualizations page) ─── */
  function initImageSlider() {
    var slider = document.getElementById("vizSlider");
    var zLabel = document.getElementById("vizZLabel");
    var epochLabel = document.getElementById("vizEpochLabel");
    var imgContainer = document.getElementById("vizImages");
    if (!slider || !imgContainer) return;

    /* Snapshot definitions — maps slider index to file and redshift */
    var snapshots = [
      { file: "z_00.png", z: 0.0 },
      { file: "z_02.png", z: 0.2 },
      { file: "z_05.png", z: 0.5 },
      { file: "z_10.png", z: 1.0 },
      { file: "z_20.png", z: 2.0 },
      { file: "z_40.png", z: 4.0 },
      { file: "z_56.png", z: 5.6 },
      { file: "z_80.png", z: 8.0 },
      { file: "z_12.png", z: 12.0 },
    ];

    /* Pre-create <img> elements */
    snapshots.forEach(function (s, i) {
      var img = document.createElement("img");
      img.src = "visualizations/" + s.file;
      img.alt = "Cosmic web at z = " + s.z;
      if (i === 0) img.classList.add("active");
      imgContainer.appendChild(img);
    });

    slider.max = snapshots.length - 1;
    slider.value = 0;

    function update() {
      var idx = parseInt(slider.value, 10);
      var s = snapshots[idx];
      imgContainer.querySelectorAll("img").forEach(function (img, i) {
        img.classList.toggle("active", i === idx);
      });
      if (zLabel) zLabel.textContent = "z = " + s.z.toFixed(1);
      if (epochLabel) {
        var epoch =
          s.z < 0.5 ? "Present" : s.z < 2 ? "Low-z" : s.z < 6 ? "Mid-z" : "High-z";
        epochLabel.textContent = epoch;
      }
    }

    slider.addEventListener("input", update);
    update();
  }

  /* ─── Initialise on DOM ready ─── */
  document.addEventListener("DOMContentLoaded", function () {
    buildNav();
    buildFooter();
    initAccordion();
    initImageSlider();
  });
})();
