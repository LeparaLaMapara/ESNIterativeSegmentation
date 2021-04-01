%% --------------------------------------------------------------------------------------------------
% Document Metadata
\def\fileversion{0.2}
\def\filedate{05 February 2017}
\typeout{Document Style Option 'WitsPhysicsReport' -- Ver. \fileversion\space Released \filedate}
\typeout{Written by Warren Carlson}

%% ----------------------------------------------------------------------------------------------------
% Supply 'WitsPhysicsReport' Class
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{WitsPhysicsReport}[2017/02/06
    v0.2
    University of the Witwatersrand, School of Physics Undergraduate Report Class]

%% Load base with options
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt, a4paper, titlepage, oneside]{article}

%% Fallback
\DeclareOption*{
    \ClassWarning{WitsPhysicsReport}{Unknown option '\CurrentOption'}
}
%% Execute default options
\ExecuteOptions{10pt}

%% Process given options
\ProcessOptions\relax

%% ----------------------------------------------------------------------------------------------------
% Table of contents formatting and structure
\usepackage{multicol}
% Formatting
\RequirePackage{tocloft}

% Add bibliography to table of contents
\RequirePackage[nottoc]{tocbibind}

\setlength\cftparskip{0.2em}
\setlength\cftbeforesecskip{1em}
\setlength\cftaftertoctitleskip{1em}

% Section numbering depths
\setcounter{secnumdepth}{4}	% Numbers section, subsection, subsubsection and paragraph in document
\setcounter{tocdepth}{3}	% Numbers section, subsection and subsubsection in Table of Contents

%% ----------------------------------------------------------------------------------------------------
% PDF Graphics
\RequirePackage{ifpdf}
\ifpdf
    \RequirePackage[pdftex]{graphicx}
    \RequirePackage{epstopdf}
\else
    \RequirePackage{graphicx}
\fi

% Document graphics
\RequirePackage[usenames,dvipsnames]{xcolor}
\RequirePackage{eso-pic}
\RequirePackage{subfig}
\RequirePackage{epsfig}

%% ----------------------------------------------------------------------------------------------------
% Dates
\RequirePackage[english,iso]{isodate}

%% ----------------------------------------------------------------------------------------------------
% Logical Convenience Packages
\RequirePackage{ifthen} % provides \ifthenelse test
\RequirePackage{xifthen} % provides \isempty test

%% ----------------------------------------------------------------------------------------------------
% Links, Citations and Referencse
\RequirePackage[%
    ocgcolorlinks, %
    linkcolor = blue, %
    urlcolor = blue, %
	citecolor = blue, %
    pageanchor = true,%
    hyperindex = true, %
    breaklinks = true, %
    pagebackref = false, %
    linktocpage = true, %
    pdfhighlight = /P]{hyperref}

% Language
\RequirePackage[latin1]{inputenc}
\RequirePackage[english]{babel}

% Page Formatting
\RequirePackage{pdflscape}
\RequirePackage{flafter}
\RequirePackage{lastpage}
\RequirePackage{pdfpages}
% \RequirePackage{fixltx2e}

\RequirePackage{fp} % Allows fixed point mathematics
\RequirePackage{caption} % Allows captions outside of floats

%% --------------------------------------------------------------------------------------------------
% Document page limit
% From: http://stackoverflow.com/questions/2720534/force-a-maximum-number-of-pages-in-latex
% Function to limit the report to 2 pages
\makeatletter
\newcounter{pagecount}
\newcommand{\limitpages}[1]{
    \setcounter{pagecount}{0}%
    \gdef\maxpages{#1}%
    \ifx\latex@outputpage\@undefined\relax%
        \global\let\latex@outputpage\@outputpage%
    \fi%
    \gdef\@outputpage{ %
        \addtocounter{pagecount}{1}%
        \ifnum\value{pagecount}>\maxpages\relax%
						\PackageError{WitsPhysicsReport}{Your specified content exceeds the page limit of the report! The extra pages have been suppressed from the output. Edit your content so that it fits}{Edit your content so that it fits}
            % Do not output the page
        \else%
            \latex@outputpage%
        \fi%
    }%
}
\makeatother

%% --------------------------------------------------------------------------------------------------
% Document Metaparameters

%% Page specification from witsthesis.sty for printing pages that won't be bound
\limitpages{1600}
\RequirePackage{fullpage}

% \RequirePackage[
%    margin=1.5cm,
%    includefoot,
%    footskip=30pt,
%    ]{geometry}


\brokenpenalty=10000
\widowpenalty=10000
\clubpenalty=10000
\interfootnotelinepenalty=10000 % Prevents footnotes being split onto two pages

\renewcommand{\textfraction}{0.0} % <--- my addition.***
\renewcommand{\topfraction}{1.0} % <--- my addition.***
\renewcommand{\bottomfraction}{1.0} % <--- my addition.***

\setcounter{topnumber}{10} % <--- my addition.***
\setcounter{bottomnumber}{10} % <--- my addition.***
\parindent 0em \parskip 2.8ex % No indents, paragraph spacing.

%% ----------------------------------------------------------------------------------------------------
% Title page information and structure

% Institution data capture
\newcommand{\School}{School of Computer Science And Applied Mathematics}
\newcommand{\University}{University of the Witwatersrand \\ \vspace*{1em} Johannesburg}
%\newcommand{\WitsCrest}{\includegraphics[width=0.35\textwidth]{WitsLogo.pdf}}
\newcommand{\WitsCrest}{\includegraphics[height=15em]{Wits_Logo.jpg}}

% Start Title page valiable scoping
\makeatletter

% Author data caputure
\gdef\@title{\textcolor{red}{PLEASE ENTER A REPORT TITILE}}
\gdef\@studentname{\textcolor{red}{PLEASE ENTER YOUR NAME}}
\gdef\@studentnumber{\textcolor{red}{PLEASE ENTER YOUR STUDENT NUMBER}}
\gdef\@studentemail{\textcolor{red}{PLEASE ENTER YOUR EMAIL ADDRESS}}
\gdef\@coursecode{\textcolor{red}{PLEASE ENTER THE COURSE CODE}}
\gdef\@coursename{\textcolor{red}{PLEASE ENTER THE COURSE NAME}}

% Author data reassignment
\newcommand{\Title}[1]        { \ifthenelse{ \isempty{#1} } {} { \gdef\@title{#1} } }
\newcommand{\CourseCode}[1]   { \ifthenelse{ \isempty{#1} } {} { \gdef\@coursecode{#1} } }
\newcommand{\CourseName}[1]   { \ifthenelse{ \isempty{#1} } {} { \gdef\@coursename{#1} } }
\newcommand{\StudentName}[1]  { \ifthenelse{ \isempty{#1} } {} { \gdef\@studentname{#1} } }
\newcommand{\StudentNumber}[1]{ \ifthenelse{ \isempty{#1} } {} { \gdef\@studentnumber{#1} } }
\newcommand{\StudentEmail}[1] { \ifthenelse{ \isempty{#1} } {} { \gdef\@studentemail{\href{mailto:#1}{#1}} } }


% Cover page, table of contents, list of figures
\newcommand{\HRule}{\rule{\linewidth}{0.5mm}}
\renewcommand{\maketitle}{ %
    \begin{titlepage}
        \begin{center}
            \textsc{\LARGE{\University}}\par
            \vspace*{1em}
            \WitsCrest\par
            \vspace*{0.5em}
            \textbf{\Large{\School}}\par
            \vspace*{0.5em}
            \textit{\Large{Thesis presented in fulfilment of the requirements for the degree of Master of Science in Computer Science at the University of the Witwatersrand}}\par
            \vspace*{0.5em}
            \HRule
            \vspace*{1em}
            \textbf{\Large{\@title}}\par
            \vspace*{1em}
            \HRule
        \end{center}
        \vspace*{0.5em}
        
        \begin{center}
                by \\
                \vspace*{0.5em}
        \Large{Thabang L. Mashinini} \\
        \end{center}
        \vspace*{0.5em}
        
         \begin{multicols}{2}
         \centering
         \Large{\textbf{Supervisor}}\\
         Prof. Terence L. Van Zyl \par
          \textbf{Co-supervisor} \\ \par
          \Large{Prof. Turgay Celik}
        \end{multicols}
        \begin{center}
            \today 
        \end{center}
        % \textbf{Student Number:}~\@studentnumber\par
        % \textbf{Email Address:}~\@studentemail\par
    \end{titlepage}
    \newpage
    \pagenumbering{roman}
    \begin{abstract}
    % Echo state networks (ESN) are a novel approach to recurrent neural network training. An ESN consist of a large, fixed recurrent \textit{reservoir} network, from which the desired output is obtained by only training suitable output weight connections. Thus the determination of optimal output weights becomes a linear task. In this thesis, the ESN is used to model the evolution of a partial differential equations (PDEs), with a particular focus on PDEs applied to image segmentation, through active contour models. And also I investigates in detail the merit of replacing the offline with the online learning approach, to the optimization of the global parameters of the ESN.
    \end{abstract}



\section*{Declaration of Authorship}
\noindent I,  THABANG LUKHETHO MASHININI, declare that this thesis titled, 'Solving Partial Differential Equations Using Reservoir Computing: An Application To Iterative Image Segmentation' and the work presented in it are my own.  I Confirm that:

\begin{itemize} 
\item This work was done wholly or mainly while in candidature for a for the degree of Masters of Science at the University of the Witwatersrand.
\item This work has not been submitted to any other institution or for any other degree.
\item Where I have consulted the published work of others, this is always clearly attributed.
\item Where I have quoted from the work of others, the source is always given. With the exception of such quotations, this research proposal is entirely my own work.
\item I have acknowledged all main sources of help.
\item Where the thesis is based on work done by myself jointly with others, I have made clear exactly what was done by others and what I have contributed myself.\\
\end{itemize}
 
\noindent \textbf{Signed}: \quad \includegraphics[height = 2cm ,width=4cm]{signature.png} \\
\rule[0.5em]{25em}{0.5pt} % This prints a line for the signature
 
\noindent \textbf{Date}: \quad \Large{\today} \\
\rule[0.5em]{25em}{0.5pt} % This prints a line to write the date

\newpage
\section*{Acknowledgements}

I Would like to thank my supervisor Prof. Terence Van Zyl for his assistance and guidance and  co-supervisor Prof. Turgay Celik for introduction me to reservoir computing, and also grateful for the funding they provided me during  towards this MSc. But Most importantly allowing me do everything myself so that I can be an independent scientist. In addition, the financial assistance of the South African Bank Seta towards this MSc, is hereby acknowledged.

I would like to thank the Mathematical Sciences Support unit for the High Performance Computing infrastructure provided for computations, and Mr Shunmuga Pillay for allowing me to push the boundaries of the computational resources provided, and letting me know that "I think you're pushing us into new territory"


%I would also like to thank Etienne Vos I have learned so much from you! Thank you very much Etienne.


Finally I would like to thank my family for their encouragement and support of everthing I do, ke ya leboga  Sekhoto sa Makolobane. Go batsadi ba me Sello le Dumazile ke leboga lorato le thego go sengwe le sengwe se ke se dirang. Go mokaulengwe le kgaitsadi Lebogang and Palesa ke leboga tshwaragano le thego e fang yona mo tseleng yaka. Ke leboge Modimo le Badimo bo Sekhoto, manelwa ke pula,  monna wa batho, makolobane, matlotlomela, matleela letlapeng.





\newpage
\section*{Acknowledgements}

I Would like to thank my supervisor Prof. Terence Van Zyl for his assistance and guidance and  co-supervisor Prof. Turgay Celik for introduction me to reservoir computing, and also grateful for the funding they provided me during  towards this MSc. But Most importantly allowing me do everything myself so that I can be an independent scientist. In addition, the financial assistance of the South African Bank Seta towards this MSc, is hereby acknowledged.

I would like to thank the Mathematical Sciences Support unit for the High Performance Computing infrastructure provided for computations, and Mr Shunmuga Pillay for allowing me to push the boundaries of the computational resources provided, and letting me know that "I think you're pushing us into new territory"


%I would also like to thank Etienne Vos I have learned so much from you! Thank you very much Etienne.


Finally I would like to thank my family for their encouragement and support of everthing I do, ke ya leboga  Sekhoto sa Makolobane. Go batsadi ba me Sello le Dumazile ke leboga lorato le thego go sengwe le sengwe se ke se dirang. Go mokaulengwe le kgaitsadi Lebogang and Palesa ke leboga tshwaragano le thego e fang yona mo tseleng yaka. Ke leboge Modimo le Badimo bo Sekhoto, manelwa ke pula,  monna wa batho, makolobane, matlotlomela, matleela letlapeng.


    \newpage
    \tableofcontents
    \newpage
    \listoffigures
    \newpage
    \listoftables
    \newpage
    \pagenumbering{arabic}
    }
    


% End Title page variable scoping
\makeatother
%% ----------------------------------------------------------------------------------------------------


%% ----------------------------------------------------------------------------------------------------
% Bibliography Style
\usepackage{natbib} 
\bibliographystyle{plain}
%\setcitestyle{authoryear,open={((},close={))}}


%% --------------------------------------------------------------------------------------------------
% Mathematics Environment and Packages
% Mathematics Packages
\RequirePackage{bm}
\RequirePackage{amsmath,amsthm,amsfonts,amssymb}

% Environments
\newtheorem{theorem}{Theorem}
\newtheorem{acknowledgement}{Acknowledgement}
\newtheorem{algorithm}{Algorithm}
\newtheorem{axiom}{Axiom}
\newtheorem{case}{Case}
\newtheorem{claim}{Claim}
\newtheorem{conclusion}{Conclusion}
\newtheorem{condition}{Condition}
\newtheorem{conjecture}{Conjecture}
\newtheorem{corollary}{Corollary}
\newtheorem{criterion}{Criterion}
\newtheorem{definition}{Definition}
\newtheorem{example}{Example}
\newtheorem{exercise}{Exercise}
\newtheorem{lemma}{Lemma}
\newtheorem{notation}{Notation}
\newtheorem{problem}{Problem}
\newtheorem{proposition}{Proposition}
\newtheorem{remark}{Remark}
\newtheorem{solution}{Solution}
\newtheorem{summary}{Summary}


%------------------------------------------------------------
% General Functions
%------------------------------------------------------------
% New definition of square root: it renames \sqrt as \oldsqrt
\let\oldsqrt\sqrt
%  it defines the new \sqrt in terms of the old one
\def\sqrt{\mathpalette\DHLhksqrt}
\def\DHLhksqrt#1#2{%
\setbox0=\hbox{$#1\oldsqrt{#2\,}$}\dimen0=\ht0
\advance\dimen0-0.2\ht0
\setbox2=\hbox{\vrule height\ht0 depth -\dimen0}%
{\box0\lower0.4pt\box2}}
% logarithm
\let\oldlog\log
\renewcommand{\log}[1]{ \ \textrm{log} \left( #1 \right) }
% natural logarithm
\let\oldln\ln
\renewcommand{\ln}[1]{ \ \textrm{ln} \left( #1 \right) }
% exponential
\let\oldexp\exp
\renewcommand{\exp}[1]{ \ \textrm{exp} \left( #1 \right) }


%--------------------------------------------------------
% Algebraic Functions
%--------------------------------------------------------
% endomorphism
\newcommand{\End}[1]{ \mathrm{End} \left( #1 \right) }
% commutator
\newcommand{\Comm}[1]{ \mathrm{Comm} \left( #1 \right) }
% representation hooks
\newcommand{\hooks}[1]{ \mathrm{hooks} \left( #1 \right) }
% representation weights
\newcommand{\weights}[2]{ \mathrm{weights}_{#2} \left( #1 \right) }
% span
\newcommand{\spanset}[1]{ \ \mathrm{span} \left\{ #1 \right\} }
% determinant
\let\olddet\det
\renewcommand{\det}[1]{ \ \mathrm{det} \left( #1 \right) }
% diagonal
\newcommand{\diag}[1]{ \ \mathrm{diag} \left( #1 \right) }
% measure
\newcommand{\dd}{\mathrm{d}}
% trace
\newcommand{\tr}[1]{ \ \textrm{tr} \left( #1 \right) }
% Residue
\newcommand{\res}[1]{ \ \textrm{Res} \left( #1 \right) }
% Real
\newcommand{\re}[1]{ \ \textrm{Re} \left( #1 \right) }
% Imaginary
\newcommand{\im}[1]{ \ \textrm{Im} \left( #1 \right) }
% sign
\newcommand{\sgn}[1]{ \ \mathrm{sgn} \left( #1 \right) }
% floor
\newcommand{\floor}[1]{ \ \lfloor #1 \rfloor }
% cieling
\newcommand{\cieling}[1]{ \ \lciel #1 \rciel }
% absolute valune
\newcommand{\abs}[1]{ \left| #1 \right| }


%--------------------------------------------------------
% Trigonometric Functions
%--------------------------------------------------------
% cosine
\let\oldcos\cos
\renewcommand{\cos}[1]{ \ \textrm{cos} \left( #1 \right) }
\newcommand{\pcos}[2]{ \ \textrm{cos}^{#2} \left( #1 \right) }
% inverse cosine
\newcommand{\acos}[1]{ \ \textrm{cos}^{-1} \left( #1 \right) }
% sine
\let\oldsin\sin
\renewcommand{\sin}[1]{ \ \textrm{sin} \left( #1 \right) }
\newcommand{\psin}[2]{ \ \textrm{sin}^{#2} \left( #1 \right) }
% inverse sine
\newcommand{\asin}[1]{ \ \textrm{sin}^{-1} \left( #1 \right) }
% tangent
\let\oldtan\tan
\renewcommand{\tan}[1]{ \ \textrm{tan} \left( #1 \right) }
\newcommand{\ptan}[2]{ \ \textrm{tan}^{#2} \left( #1 \right) }
% inverse tangent
\newcommand{\atan}[1]{ \ \textrm{tan}^{-1} \left( #1 \right) }

% hyperbolic cosine
\let\oldcosh\cosh
\renewcommand{\cosh}[1]{ \ \textrm{cosh} \left( #1 \right) }
\newcommand{\pcosh}[2]{ \ \textrm{cosh}^{#2} \left( #1 \right) }
% hyperbolic sine
\let\oldsinh\sinh
\renewcommand{\sinh}[1]{ \ \textrm{sinh} \left( #1 \right) }
\newcommand{\psinh}[2]{ \ \textrm{sinh}^{#2} \left( #1 \right) }
% hyperbolic tangent
\let\oldtanh\tanh
\renewcommand{\tanh}[1]{ \ \textrm{tanh} \left( #1 \right) }
\newcommand{\ptanh}[2]{ \ \textrm{tanh}^{#2} \left( #1 \right) }


%--------------------------------------------------------
% Dirac Notation
%--------------------------------------------------------
% bra
\newcommand{\bra}[1]{ \left< \left. #1 \right. \right| }
% ket
\newcommand{\ket}[1]{ \left| \left. #1 \right. \right> }
% expectation value
\newcommand{\expect}[1]{ \left< #1 \right> }
% braket
\newcommand{\braket}[2]{ \left< \left. #1 \right| #2 \right> }
% dimension
\let\olddim\dim
\renewcommand{\dim}[1]{ \mathrm{dim} \left( #1 \right) }
\newcommand{\ndim}[2]{ \mathrm{dim}_{ #1 } \left( #2 \right) }


%--------------------------------------------------------
% derivatives
%--------------------------------------------------------
% partial operator
\newcommand{\pdiffop}[1]{ \frac{ \partial }{ \partial {#1} } }
\newcommand{\ppdiffop}[2]{ \frac{ \partial^{#2} }{ \partial {#1}^{#2} } }
% partial
\newcommand{\pdiff}[2]{ \left( \frac{ \partial #1 }{ \partial #2 } \right) }
\newcommand{\ppdiff}[3]{ \left( \frac{ \partial^{#3} #1 }{ \partial {#2}^{#3} } \right) }
% total operator
\newcommand{\tdiffop}[1]{ \frac{ \dd }{ \dd #1 } }
\newcommand{\ptdiffop}[2]{ \frac{ \dd^{#2} }{\dd {#1}^{#2} } }
% total
\newcommand{\tdiff}[2]{ \left( \frac{ \dd #1 }{ \dd #2 } \right) }
\newcommand{\ptdiff}[3]{ \left( \frac{ \dd^{#3} #1 }{ \dd {#2}^{#3}} \right) }
%-------------------------------------------------------


%% ----------
\endinput

%% --------------------------------------------------
%% EOF
