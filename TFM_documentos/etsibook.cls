%%%%%%% IMPORTANTE: NO EDITE Y REDISTRIBUYA ESTE DOCUMENTO, SE HA GENERADO AUTOMÁTICAMENTE %%%%%%%

%% ----- CARGA DE LA CLASE BASE Y OPCIONES ----- %%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{etsibook}[2026/04/29 Plantilla de publicaciones de la ETSi (US)]
% Opciones del paquete
\RequirePackage{kvoptions}
\RequirePackage{etoolbox}
\SetupKeyvalOptions{
   family=etsi,
   prefix=etsi@,
}

% Redirección de opciones a la clase padre
\DeclareStringOption[10pt]{fontsize}
\DeclareStringOption[a4]{paper}
\DeclareStringOption[twoside]{sides}
\DeclareStringOption[project]{mode}

% Procesar opciones
\ProcessKeyvalOptions*

% Comprueba si es project o book
\ifdefstring{\etsi@mode}{project}{}{
  \ifdefstring{\etsi@mode}{book}{}{
    \ifdefstring{\etsi@mode}{thesis}{}{
      \ClassError{etsitex}{Opción 'mode': \etsi@mode}{Opciones: 'project' (TFG/TFM), 'book' o 'thesis'.}
    }
  }
}
\typeout{etsitex: modo \etsi@mode detectado}

% Pasar opciones relevantes 
\PassOptionsToClass{fontsize=\etsi@fontsize}{scrbook}
\PassOptionsToClass{paper=\etsi@paper}{scrbook}
\PassOptionsToClass{\etsi@sides}{scrbook}
\PassOptionsToClass{numbers=noenddot}{scrbook}
\PassOptionsToClass{listof=totoc}{scrbook}
\PassOptionsToClass{bibliography=totoc}{scrbook}
% Cargar clase base
\LoadClass{scrbook}

\RequirePackage{ifdraft}

%% ----- PERSONALIZACIÓN DE LA CLASE (KOMA SCRIPT) ----- %%
\makeatletter
%% DEFINICIONES
\newcommand{\etsilogo}{etsi/logo-etsi}
\newcommand{\coverimg}{etsi/imagen-libro}
\newcommand{\uslogo}{etsi/logo-us}
\newcommand{\phlogo}{etsi/logo-dpto-placeholder}
% Definiciones para modo TFG/TFM
\ifdefstring{\etsi@mode}{project}{
    \newcommand{\coverbg}{etsi/portada-tfc.pdf}
}{}
% Definiciones para modo Libro
\ifdefstring{\etsi@mode}{book}{
    \newcommand{\coverbg}{etsi/portada-libro.pdf}
}{}
% Definiciones para modo Tesis
\ifdefstring{\etsi@mode}{thesis}{
    \newcommand{\coverbg}{etsi/portada-libro.pdf}
}{}
%% AJUSTES DE LA DISPOSICIÓN Y GEOMETRÍA DE PÁGINA
\ifdefstring{\etsi@paper}{a4}{%
  \KOMAoptions{
    DIV=12,
    BCOR=10mm,
  }
}{
  \ifdefstring{\etsi@paper}{b5}{%
    \KOMAoptions{
      DIV=12,
      BCOR=5mm,
    }
  }{
    \ifdefstring{\etsi@paper}{a5}{%
    \KOMAoptions{
      DIV=12,
      BCOR=5mm,
    }
  }{
      \KOMAoptions{
        DIV=12,
      }
    }
  }
}

%% AJUSTES DE METADATOS DEL DOCUMENTO
%% Cabecera de la portada
\newcommand{\header}[1]{\gdef\@header{#1}}
\newcommand{\@header}{\@latex@warning@no@line{No \noexpand\header given}}

%% Texto de afiliación y ubicación
\newcommand{\affiliation}[1]{\gdef\@affiliation{#1}}
\newcommand{\@affiliation}{\@latex@warning@no@line{No \noexpand\affiliation given}}

%% Cabecera de la portada
\newcommand{\authorlabel}[1]{\gdef\@authorlabel{#1}}
\newcommand{\@authorlabel}{Autor}  % Valor por defecto

%% Autores y directores
\newcommand{\@director}{}  % Inicializar vacío
\newcommand{\@directordetails}{}  % Inicializar vacío
\newcommand{\director}[3]{%
  \expandafter\gdef\expandafter\@director\expandafter{\@director #1: #2\\}
  \expandafter\gdef\expandafter\@directordetails\expandafter{%
    \@directordetails

    \vspace{0.1cm}
    
    #1:
    
    {\large\let\smaller\relax#2}
    
    #3%
  }%
}

% Logo del Departamento
\newcommand{\optlogo}[1]{\gdef\@optlogo{#1}}
\newcommand{\@optlogo}{unspecified}  % Valor por defecto


%% CONFIGURACIÓN DE FUENTES
\RequirePackage[T1]{fontenc}
\RequirePackage{relsize}
% Fuentes utilizadas
\renewcommand*{\ttdefault}{pcr}
\renewcommand*{\sfdefault}{phv}
\renewcommand*{\rmdefault}{qtm}
\newcommand{\useheaderfont}{\usefont{T1}{phv}{bc}{n}}
\newcommand{\uselabelfont}{\usefont{T1}{phv}{mc}{n}}
\newcommand{\uselightfont}{\usefont{T1}{phv}{c}{n}}
\newcommand{\usenormalfont}{\rmfamily}
\renewcommand{\RSlargest}{30pt}
% Capítulos (2.4x normal)
\setkomafont{chapter}{\Huge\useheaderfont\selectfont}
\setkomafont{chapterprefix}{\Huge\useheaderfont\selectfont}
% Partes (2.8x normal)  
\setkomafont{part}{\larger\Huge\useheaderfont\selectfont}
\setkomafont{partprefix}{\larger\Huge\useheaderfont\selectfont}
% Secciones (1.3x normal)
\setkomafont{section}{\relsize{1.3}\useheaderfont\selectfont}
% Subsecciones (normal)
\setkomafont{subsection}{\normalsize\useheaderfont\selectfont}
\setkomafont{subsubsection}{\normalsize\useheaderfont\selectfont}
\setkomafont{paragraph}{\normalsize\useheaderfont\selectfont}
\setkomafont{subparagraph}{\normalsize\useheaderfont\selectfont}
% Cabeceras y pie de página (0.9x normal)
\setkomafont{pageheadfoot}{\small\useheaderfont\selectfont}
\setkomafont{pagenumber}{\small\useheaderfont\selectfont}
% Tabla de contenidos 
\setkomafont{partentry}{\relsize{1.8}\useheaderfont\selectfont}
\setkomafont{chapterentry}{\relsize{1.1}\useheaderfont\selectfont}
\DeclareTOCStyleEntry[
  level=1,
  indent=1.5em,
  numwidth=2.3em,
  entryformat=\normalsize\uselabelfont\selectfont,
  pagenumberformat=\normalsize\uselabelfont\selectfont
]{tocline}{section}
\DeclareTOCStyleEntry[
  level=2,
  indent=3.8em,
  numwidth=3.2em,
  entryformat=\normalsize\uselabelfont\selectfont,
  pagenumberformat=\normalsize\uselabelfont\selectfont
]{tocline}{subsection}
% Títulos de documento 
\ifdefstring{\etsi@mode}{project}{%
\setkomafont{title}{\Huge\usenormalfont\hyphenpenalty=10000}
\setkomafont{subtitle}{\Large\usenormalfont\hyphenpenalty=10000}
\newkomafont{coverheader}{\Huge\usenormalfont\hyphenpenalty=10000}
\setkomafont{author}{\Large\usenormalfont\hyphenpenalty=10000}
\setkomafont{date}{\normalsize\usenormalfont\hyphenpenalty=10000}
\newkomafont{affiliation}{\Large\usenormalfont\bfseries\hyphenpenalty=10000}
}{}
\ifdefstring{\etsi@mode}{book}{%
\setkomafont{title}{\Huge\useheaderfont\selectfont\hyphenpenalty=10000}
\setkomafont{subtitle}{\huge\useheaderfont\selectfont\hyphenpenalty=10000}
\newkomafont{coverheader}{\huge\useheaderfont\selectfont\hyphenpenalty=10000}
\setkomafont{author}{\LARGE\useheaderfont\selectfont\hyphenpenalty=10000}
\setkomafont{date}{\normalsize\useheaderfont\selectfont\hyphenpenalty=10000}
\newkomafont{affiliation}{\large\uselightfont\selectfont\hyphenpenalty=10000}
}{}
\ifdefstring{\etsi@mode}{thesis}{%
\setkomafont{title}{\Huge\useheaderfont\selectfont\hyphenpenalty=10000}
\setkomafont{subtitle}{\huge\useheaderfont\selectfont\hyphenpenalty=10000}
\newkomafont{coverheader}{\huge\useheaderfont\selectfont\hyphenpenalty=10000}
\setkomafont{author}{\LARGE\useheaderfont\selectfont\hyphenpenalty=10000}
\setkomafont{date}{\normalsize\useheaderfont\selectfont\hyphenpenalty=10000}
\newkomafont{affiliation}{\LARGE\uselightfont\selectfont\hyphenpenalty=10000}
}{}
% Descripciones
\setkomafont{descriptionlabel}{\normalsize\useheaderfont\selectfont}
\setkomafont{captionlabel}{\normalsize\useheaderfont\selectfont}
\setkomafont{caption}{\normalsize\usenormalfont\selectfont}
% Texttt
\let\oldtexttt\texttt
\renewcommand{\texttt}[1]{{\smaller{}\oldtexttt{#1}}}

%% DEFINICIÓN DE LA PORTADA
\RequirePackage[draft=false]{scrlayer-scrpage}
\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{setspace}
% Fondo para la portada
\DeclareNewLayer[
  background,
  mode=picture,
  contents={%
    \includegraphics[width=\layerwidth,height=\layerheight]{\coverbg}%
  }
]{etsi.titlepage.bg}
\DeclareNewPageStyleByLayers{etsi.titlepage}{etsi.titlepage.bg}
% Construcción de la portada
\RequirePackage[absolute,overlay]{textpos}
\RequirePackage{ragged2e}
\ifdefstring{\etsi@mode}{project}{
  \newcommand{\makeetsicover}{%
    \setlength{\TPHorizModule}{0.01\paperwidth}
    \setlength{\TPVertModule}{0.01\paperheight}
    \begin{titlepage}%
      \thispagestyle{etsi.titlepage}
      \begin{textblock}{60}(20, 7)
        \noindent\usekomafont{coverheader}\@header\par
      \end{textblock}%
      \begin{textblock}{60}(20, 40)
        \justify\noindent\usekomafont{title}\@title\par
      \end{textblock}%
      \ifx\@author\@empty
        % no author
      \else
        \begin{textblock}{48}(20, 55)
          \noindent\usekomafont{author}\@authorlabel: \@author\\\@director\par
        \end{textblock}%
      \fi
      \begin{textblock}{100}(0, 74)
        \centering\noindent%
        {
          \color{white}
          \usekomafont{affiliation}\@affiliation%
          \\[1.5em]%
          \usekomafont{date}\@date\par
        }
      \end{textblock}%
      \begin{textblock}{27}(80, 73)
        \includegraphics{\etsilogo}
      \end{textblock}%
      \begin{textblock}{100}(0, 90)
        \centering%
        \includegraphics[width=0.12\paperwidth]{\uslogo}%
        \IfFileExists{\@optlogo}{%
          \hspace{0.048\paperwidth}%
          \includegraphics[width=0.11\paperwidth]{\@optlogo}%
        }{%
          \ClassWarning{etsi}{Optional logo file '\@optlogo' not found}%
        }%
      \end{textblock}%

    \vspace*{\fill}%
    \end{titlepage}%
  }
}{}

\ifdefstring{\etsi@mode}{book}{
  \newcommand{\makeetsicover}{%
  \setlength{\TPHorizModule}{0.01\paperwidth}%
  \setlength{\TPVertModule}{0.01\paperheight}%
    \begin{titlepage}%
      \thispagestyle{etsi.titlepage}
      \begin{textblock}{56}(25, 9.5)%
        {%
          \color[RGB]{203,148,121}%
          \raggedleft\noindent\usekomafont{title}\relsize{1}\@title
          \\[1em]%
          \hfill\usekomafont{subtitle}\@subtitle\par%
        }%
      \end{textblock}%
      \begin{textblock}{51.5}(28, 31.8)%
        \includegraphics[width=51.5\TPHorizModule]{\coverimg}%
      \end{textblock}%
      \begin{textblock}{58}(20, 74.5)%
        \raggedleft\noindent%
        {%
          \color[RGB]{203,148,121}%
          \usekomafont{author}\@author%
          \\[.2em]%
          \usekomafont{affiliation}\@affiliation\par%
        }%
      \end{textblock}%
      \begin{textblock}{27}(80, 73)%
        \includegraphics[height=.11\paperheight]{\etsilogo}%
      \end{textblock}%

      \vspace*{\fill}%
    \end{titlepage}%
  }
  \newcommand{\makecopyright}[3]{%
    \begin{titlepage}%
      \thispagestyle{empty}
      \usekomafont{affiliation}\let\smaller\relax
      \noindent\large \textcopyright{} \@date. \@author\\[1em]
      \noindent\textcopyright{} Escuela Técnica Superior de Ingeniería\\
      Camino de los Descubrimientos sn\\
      41092, Sevilla\\
      www.etsi.us.es\\
      Sevilla, \@date 
      \\[1em]
      Impreso en España - Printed in Spain\\
      Registro de la propiedad intelectual\hspace{2em}{#1}\\
      ISBN\hspace{2em}{#2}\\

      \noindent#3

      \vspace*{\fill}
      
      \noindent{}No está permitida la distribución de este texto, ni el código asociado, a terceras personas, por cualquier medio. El uso de este texto y el código asociado está restringido al Servicio de Publicaciones de la Universidad de Sevilla, así como a los proyectos final de carrera, máster, tesis y documentos similares que se defiendan en la misma.\\
      
      \vspace*{\fill}
      
      \noindent{}Diseño de cubierta: Fernando García García.\\
      Imagen central de cubierta: Postal de Maxwell a Tait.\\
      Diseño de maquetación (\LaTeX): F. Javier Payán Somet \textcopyright{} 2014. 

      \vspace*{\fill}

    \end{titlepage}%
  }
  \newcommand{\maketitlebackcover}{%
    \begin{titlepage}%
      \thispagestyle{empty}
      
      \raggedleft\noindent
      \begin{textblock}{56}(25, 15)%
        \raggedleft\noindent\usekomafont{title}\relsize{1}\@title
        \\[1em]%
        \hfill\usekomafont{subtitle}\@subtitle\par%
        \vspace{-0.5em}
        {\noindent\color{gray!75}\rule{\textwidth}{3.5pt}}
      \end{textblock}
      
      \vspace*{\fill}

    \end{titlepage}%
  }
  \newcommand{\makeauthorbackcover}{%
    \begin{titlepage}%
      \thispagestyle{empty}
      
      \raggedleft\noindent
      \begin{textblock}{56}(25, 15)%
        \raggedleft\noindent\usekomafont{title}\relsize{1}\@title
        \\[1em]%
        \hfill\usekomafont{subtitle}\@subtitle\par%
        \vspace{-0.5em}
        {\noindent\color{gray!75}\rule{\textwidth}{3.5pt}}
      \end{textblock}
      
      \begin{textblock}{58}(24, 50)%
        \raggedleft\noindent%
        {%
          \usekomafont{author}\@author%
          \\[.2em]%
          \usekomafont{affiliation}\@affiliation\par%
        }%
      \end{textblock}%

      \vspace*{\fill}

    \end{titlepage}%
  }
}{
  \newcommand{\makerawcover}{%
    \begin{titlepage}%
      \thispagestyle{empty}
      \begin{center}
        {\Large\let\smaller\relax\@header\par}
        
        \vspace*{\fill}

        {\huge\bfseries\let\smaller\relax\@title\par}
        
        \vspace*{\fill}

        {\@authorlabel:
        
        {\large\let\smaller\relax\@author\par}

        \vspace{2em}
        
        \@directordetails\par}
        
        \vspace*{\fill}
        
        {\Large\let\smaller\relax\@affiliation\par}
        
        \vspace{1em}
        
        {\@date\par}
      \end{center}
      
    \end{titlepage}%
  }
}

\ifdefstring{\etsi@mode}{thesis}{
  \newcommand{\makeetsicover}{%
  \setlength{\TPHorizModule}{0.01\paperwidth}%
  \setlength{\TPVertModule}{0.01\paperheight}%
    \begin{titlepage}%
      \thispagestyle{etsi.titlepage}
      \raggedright\noindent
      \begin{textblock}{60}(25, 5)%
        {%
          \color[RGB]{203,148,121}%
          \usekomafont{coverheader}\relsize{1}{\@header}
          \\[1em]
          \usekomafont{title}\relsize{1}{\@title}\par
        }%
      \end{textblock}%
      \begin{textblock}{51.5}(25, 31.8)%
        \includegraphics[width=51.5\TPHorizModule]{\coverimg}%
      \end{textblock}%
      \begin{textblock}{58}(25, 62)%
        %
        {%
          \color[RGB]{203,148,121}%
          \usekomafont{author}\@authorlabel: \@author\\[0.1em]\@director\par%
        }%
      \end{textblock}%
      \begin{textblock}{58}(25, 74.5)%
        %
        {%
          \color[RGB]{203,148,121}%
          \usekomafont{affiliation}{\@affiliation}\par%
        }%
      \end{textblock}%
      \begin{textblock}{27}(80, 73)%
        \includegraphics[height=.11\paperheight]{\etsilogo}%
      \end{textblock}%

      \vspace*{\fill}%
    \end{titlepage}%
  }
}{}



\ifdefstring{\etsi@mode}{book}{
  \renewcommand{\maketitle}{\makeetsicover\maketitlebackcover\makeauthorbackcover}
}{
  \renewcommand{\maketitle}{\makeetsicover\makerawcover}
}


%% ESPACIADO Y SALTOS DE PÁGINA
%% Ajustes de espacio antes/después de capítulos
\RedeclareSectionCommand[
  beforeskip=75pt,
  afterskip=50pt
]{chapter}
%% Estilo de los capítulos (línea gris)
\renewcommand*{\chapterlinesformat}[3]{%
  \if\relax\detokenize{#2}\relax%
      % Capítulos sin número
      {\raggedleft#3\par}%
      {\color{gray!75}\rule{\textwidth}{3.5pt}}%
  \else%
      % Capítulos con número
      \@hangfrom{#2}{#3}%
      \par\nobreak
      {\color{gray!75}\rule{\textwidth}{3.5pt}}%
  \fi%
}
\setcounter{secnumdepth}{3} 



%% NUMERACIÓN DE CAPÍTULOS Y SECCIONES
% Configurar profundidad de numeración (hasta subsubsection)
% Configurar numeración para frontmatter (preíndice)
\renewcommand{\frontmatter}{%
  \cleardoublepage
  \@mainmatterfalse
  \pagenumbering{Roman}% Numeración romana mayúscula (I, II, III, ...)
  \setcounter{page}{1}% Empezar desde I
}
% Configurar numeración para mainmatter (contenido principal)
\renewcommand{\mainmatter}{%
  \cleardoublepage
  \@mainmattertrue
  \pagenumbering{arabic}% Numeración numérica
  \setcounter{page}{1}% Empezar desde 1
}
% Configurar numeración para backmatter (apéndices, bibliografía, etc.)
\renewcommand{\backmatter}{%
  \@mainmattertrue
}
% Saltos de página automáticos
\KOMAoptions{open=right}



\clearpairofpagestyles

%% FORMATO DE CABECERA Y PIE DE PÁGINA% Color de la regla del header
\addtokomafont{headsepline}{\color{gray!75}}
%% Cabeceras
\lehead{\pagemark\quad\rightmark}
\rohead{\leftmark\quad\pagemark}
\KOMAoptions{headsepline=.4pt}
% Marcas
\renewcommand{\chaptermarkformat}{\chapapp~\thechapter.\enskip}
\renewcommand{\sectionmarkformat}{\thesection\enskip}
\automark[chapter]{section} 
% Primera página de un capítulo
\newpairofpagestyles{chapterplain}{
  \KOMAoptions{headsepline=0pt}\cfoot{\pagemark}
}
\renewcommand*{\chapterpagestyle}{chapterplain}%% NUMERACIÓN Y TABLAS DE CONTENIDO
\DeclareTOCStyleEntry[linefill=\hfill]{tocline}{chapter}
\DeclareTOCStyleEntry[linefill=\hfill]{tocline}{section}
\DeclareTOCStyleEntry[linefill=\hfill]{tocline}{subsection}
\DeclareTOCStyleEntry[dynnumwidth,entrynumberformat=\useheaderfont\figurename~\autodot,linefill=\hfill,entryformat=\uselabelfont,pagenumberformat=\useheaderfont]{tocline}{figure}
\DeclareTOCStyleEntry[dynnumwidth,entrynumberformat=\useheaderfont\tablename~\autodot,linefill=\hfill,entryformat=\uselabelfont,pagenumberformat=\useheaderfont]{tocline}{table}
\appto\appendix{%
  % Sobrescribe el formato del contador (solo para los apéndices)
  \renewcommand*{\thechapter}{\Alph{chapter}}
}
% Configuración de tablas que dependen de los paquetes cargados
\AtBeginDocument{
  \@ifpackageloaded{listings}{%
    \DeclareTOCStyleEntry[dynnumwidth,entrynumberformat=\useheaderfont\lstlistingname~\autodot,linefill=\hfill,entryformat=\uselabelfont,pagenumberformat=\useheaderfont]{tocline}{lstlisting}%
  }{}
}%% ENTORNOS
% Epígrafes
\newlength{\myepigraphwidth}
\setlength{\myepigraphwidth}{8cm} % ancho máximo opcional
\newlength{\myepigruleheight}
\setlength{\myepigruleheight}{0pt} % altura de línea horizontal opcional
\newenvironment{epigraph}[2]{%
  \vspace*{2em} 
  \begin{flushright}
    {\parbox{\myepigraphwidth}{%
      \flushright
      \itshape #1
      \ifdim\myepigruleheight>0pt
        \hrule height \myepigruleheight
      \else
        \vspace*{1ex}
      \fi
    }}
    
    \vspace{0.1em}
    
    {\parbox{\myepigraphwidth}{%
      \flushright
      \scshape #2
    }}
  \end{flushright}
  \vspace{2em} 
}{}
%% PERSONALIZACIÓN SEGÚN PAQUETES
\AtBeginDocument{
  % Código para todos los idiomas
  \@ifpackageloaded{listings}{%
      \lstset{%
        flexiblecolumns=true,%
        backgroundcolor=\color{black!10},%
        tabsize=2,%
        aboveskip=12pt,%
        belowcaptionskip=7pt,%
        showstringspaces=false,%
        basicstyle=\smaller{}\ttfamily,%
        keywordstyle=\color{blue},%
        commentstyle=\color{gray},%
        stringstyle=\color{red},%
        captionpos=t,%
        frame=top,%
        frame=bottom,%
        framerule=0.4pt,%
        xleftmargin=6pt,%
        framexleftmargin=6pt,%
        xrightmargin=6pt,%
        framexrightmargin=6pt,%
        breaklines=true%
      }%
      \lstset{literate=%
          {á}{{\'a}}1%
          {é}{{\'e}}1%
          {í}{{\'i}}1%
          {ó}{{\'o}}1%
          {ú}{{\'u}}1%
          {Á}{{\'A}}1%
          {É}{{\'E}}1%
          {Í}{{\'I}}1%
          {Ó}{{\'O}}1%
          {Ú}{{\'U}}1%
          {ñ}{{\~{n}}}1%
          {º}{{\textsuperscript{\b{o}}}}1%
          {ª}{{\textsuperscript{\b{a}}}}1%
          {¿}{{?`}}1%
        }%
  }{}%
}%% TRADUCCIONES DE ETIQUETAS
\AtBeginDocument{
  \@ifpackageloaded{babel}{%
    \ifdefstring{\languagename}{nil}{%
      \selectlanguage{spanish}%
      \edef\CurrentBabelLanguage{spanish}%
    }{%
      \edef\CurrentBabelLanguage{\languagename}%
    }%
    \ifdefstring{\CurrentBabelLanguage}{spanish}{%
      \addto\extrasspanish{%
          \renewcommand*{\contentsname}{Índice}%
          \renewcommand*{\listfigurename}{Índice de Figuras}%
          \renewcommand*{\listtablename}{Índice de Tablas}%
          \renewcommand*{\tablename}{Tabla}%
      }
      \@ifpackageloaded{listings}{%
        \renewcommand*{\lstlistlistingname}{Índice de Códigos}%
        \renewcommand*{\lstlistingname}{Código}%
      }{}%
      \@ifpackageloaded{glossaries}{%
         \setglossarypreamble[acronym]{\addcontentsline{toc}{chapter}{Glosario}}%
         \addto\extrasspanish{%
            \renewcommand*{\acronymname}{Glosario}%
         }
      }{}%
    }{%
      % Código para otros idiomas (opcional)
      \@ifpackageloaded{glossaries}{%
         \setglossarypreamble[acronym]{\addcontentsline{toc}{chapter}{Glossaries}}%
      }%
    }%
  }{}%
}\makeatother
\endinput
