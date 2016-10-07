<!DOCTYPE html>
<html lang="pt">
  <head>
  <!--
CAROLINE BOMFIM DO ESPIRITO SANTO RA: 20371672
KAIO AUGUSTO VITORINO RA: 20591257
RALDNEY ALVES SAMPAIO RA: 20203136
LUCAS DA SILVA RA: 20511359
-->
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no" />
    <title>Detalhes | Enershoes</title>
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <link rel="stylesheet" type="text/css" href="http://localhost/Enershoes/lib/fonts/css/font-awesome.css">
    <link type="text/css" rel="stylesheet" href="http://localhost/Enershoes/lib/materialize/css/materialize.min.css"  media="screen,projection"/>
    <link rel="stylesheet" type="text/css" href="http://localhost/Enershoes/css/style.css">

      <link rel="stylesheet" type="text/css" href="bower_components/toastr/toastr.css">
  </head>
  <body >
    <div class="navbar-fixed">
      <nav class="black " role="navigation">
        <div class="container">
          <div class="nav-wrapper">
            <div id="logo-container">
              <a href="http://localhost/Enershoes/index.php"><img width="40" class="icon hide-on-med-and-down" src="images/raio_circ2.png" alt=""></a>
              <a href="http://localhost/Enershoes/index.php" class="brand-logo">enershoes</a>
            </div>
            <div class="menu-container">
              <ul id="menu" class="right">
                <li><a href="http://localhost/Enershoes/sobre.php">Sobre</a>
                </li>
                <li ><a href="#">Masculino</a>
                </li>
                <li ><a href="#">Feminino</a>
                </li>
                <li ><a href="http://localhost/Enershoes/cadastro.php">Login/Cadastre-se</a>
                </li>
                <li ><a href="#"><i class="fa fa-shopping-cart" aria-hidden="true"></i>
                  </a>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </nav>
    </div>
    <div class="container">
      <div class="section">
        <div class="row">
          <div class="col l12">
            <article class="post-6583 post type-post status-publish format-standard has-post-thumbnail hentry category-westmount" id="post-6583">
              <div class="row">
                <div class="col l8 s12">
                  <h2>Enershoes Feminino</h2>
                  <img src="http://localhost/Enershoes/images/Feminino/Feminino3" class="responsive-img" alt="">
                  <div class="row" id="instagram-photos"></div>
                </div>
                <div class="col l4 s12">
                  <div class="card-panel alcaramel" style="min-height: 640px;">
                    <h6>Detalhes</h6>
                    <hr>
                    <span class="detail-title"></i>Nivel de carga:</span>
                    <span class="">3000mAh/KM</span>
                    <hr>
                    <span class="detail-title"><i class="mdi-action-store"></i>Sobre:</span>
                    <span class="">Tênis ótimo para corridas e carga de dispositivos</span>
                    <hr>
                    <span class="detail-title"><i class="mdi-device-access-time"></i> Disponibilidade:</span>
                    <ul class="opening-hours">
                      <li>Em breve...</li>
                    </ul>
                    <hr>
                    <a class="waves-effect waves-light btn modal-trigger" href="#edit-modal">Solicitar produto</a>
                    <div id="edit-modal" class="modal bottom-sheet modal-fixed-footer">
                      <form action="">
                        <div class="modal-content">
                          <div class="row">
                            <div class="col s12">
                              <h4>
                                Solicitar produto!
                              </h4>
                            </div>
                            <div class="input-field col s12">
                              <textarea id="edit_text" name="edit_text" rows="20" class="materialize-textarea"></textarea>
                              <label for="edit_text">Diga o que espera do nosso produto.</label>
                            </div>
                            <div class="input-field col s12">
                              <input id="edit_email" name="edit_email" type="email" class="validate" value="">
                              <label for="edit_email">Seu e-mail:</label>
                            </div>
                          </div>
                        </div>
                        <div class="modal-footer">
                           <button type="submit" onclick="toastr.success('Solicitação enviada com sucesso');" name="action" value="edit_shop" class="btn teal">
                          Submit
                          </button>
                        </div>
                      </form>
                    </div>
                  </div>
                </div>
              </div>
            </article>
          </div>
        </div>
      </div>
    </div>
    <footer class="page-footer black-footer">
      <div class="container">
        <div class="row">
          <div class="col l3 s12 m6">
            <article id="text-4" >
              <h5 class="black-text">Sobre nós</h5>
              <div>Marketing e Ciencia top da computação</div>
            </article>
          </div>
          <div class="col l3 s12 m6">
            <article >
              <h5 class="black-text">Categorias</h5>
            </article>
          </div>
          <div class="col l3 s12 m6">
            <article id="text-5" class="panel widget widget_text">
              <h5 class="black-text">Social</h5>
              <div>
                <div class="social-icons">
                  <ul>
                  </ul>
                </div>
              </div>
            </article>
          </div>
        </div>
      </div>
      <div class="footer-copyright">
        <div class="container">
          Feito em São Paulo por <a class="white-text text-lighten-4" href="http://localhost/enershoes">Raldney</a>
        </div>
      </div>
    </footer>
    <script type="text/javascript" src="https://code.jquery.com/jquery-2.1.1.min.js"></script>
    <script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/angularjs/1.5.8/angular.min.js"></script>
    <script type="text/javascript" src='bower_components/toastr/toastr.js'></script>
    <script type="text/javascript" src="http://localhost/Enershoes/lib/materialize/js/materialize.min.js"></script>
      <script type="text/javascript">


         var xValue, yValue;
         var isYIncreasing;
         $('html, body').mousemove(function(event) {
             var isYIncreasing = yValue > event.pageY;
             xValue = event.pageX;
             yValue = event.pageY;
             if (isYIncreasing && event.pageY < ($(window).scrollTop() + 25)) {
                 if ($.cookie('modal_shown') == null) {
                     $.cookie('modal_shown', 'yes', {
                         expires: 3,
                         path: '/'
                     });
                     $('#modal1').openModal();
                     analytics.track('Modal Shown', {
                         category: 'Newsletter',
                         label: 'New Shops',
                         value: 1
                     });
                 }
             }
         });

         $('#search-button').click(function() {
             $('.search').slideDown(500).focus();
         });
         $('.datepicker').pickadate({
             selectMonths: true,
             selectYears: 15
         });
         $('.offer-price-trigger').on('change', function() {
             var amount = $(this).data('value');
             var multiplier = $(this).prop('checked') ? 1 : -1;
             var cost = $(this).parents('form').find('.cost');
             var current = cost.text();
             var new_cost = current * 1 + (multiplier * amount);
             cost.text(new_cost);
         });
         $(document).ready(function() {
             $('select').material_select();
             $('.modal-trigger').leanModal();
                 var edit_modal = $('#edit-modal');
                 if (edit_modal.length > 0) {
                     var edit_text = edit_modal.find('textarea');
                     edit_modal.find('.edit-options').find('a').click(function(e) {
                         e.preventDefault();
                         var text;
                         if (typeof $(this).attr('data-text') !== 'undefined') {
                             text = $(this).attr('data-text');
                         } else {
                             text = $(this).text();
                         }
                         if (text.length > 0) {
                             text += '. ';
                         }
                         edit_text.val(text);
                         edit_text.trigger('change');
                         edit_text.focus();
                         if (typeof edit_text[0].selectionStart == "number") {
                             edit_text[0].selectionStart = edit_text[0].selectionEnd = edit_text[0].value.length;
                         } else if (typeof edit_text[0].createTextRange != "undefined") {
                             edit_text[0].focus();
                             var range = edit_text[0].createTextRange();
                             range.collapse(false);
                             range.select();
                         }
                     });
                 }
         });
      </script>
  </body>
</html>

