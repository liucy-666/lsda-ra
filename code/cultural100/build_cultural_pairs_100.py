from __future__ import annotations

import json
import re
from pathlib import Path


OUT = Path(r"D:\Python\MMDIT\AAA_Experiment\cultural_pairs_100.json")
SHELL = (
    "Neutral studio background: {A} on the left, {B} on the right; "
    "both fully visible, separate, and similar in size."
)


# Each tuple is: A short, A appended diagnostic description, B short, B appended
# diagnostic description. A is the provisionally broader-recognition anchor and B
# the provisionally narrower-recognition candidate. Model-effective familiarity
# must still be measured from standalone Short generations.
PAIRS = [
    # 1-10: porcelain and fine ceramics
    (
        "a Chinese blue-and-white porcelain vase",
        "with cobalt-blue underglaze painting beneath a transparent glaze, showing Chinese landscape, lotus, and scrolling-cloud motifs",
        "a Vietnamese Bát Tràng blue-and-white ceramic vase",
        "with cobalt-blue hand-painted decoration on a pale glazed body, showing stylized lotus, fish, and dense scrolling foliage",
    ),
    (
        "a Chinese blue-and-white porcelain plate",
        "with cobalt-blue underglaze painting beneath a transparent glaze, showing Chinese landscape, lotus, and scrolling-cloud motifs",
        "a Japanese Nabeshima porcelain plate",
        "with refined underglaze-blue outlines and sparse overglaze color, showing asymmetrical flowers, textiles, and geometric border motifs",
    ),
    (
        "a Chinese blue-and-white porcelain vase",
        "with cobalt-blue underglaze painting beneath a transparent glaze, showing Chinese landscape, lotus, and scrolling-cloud motifs",
        "a Thai Bencharong porcelain vase",
        "with dense polychrome overglaze enamel and gold outlines, showing repeating flame, deity, and floral motifs on a dark ground",
    ),
    (
        "a Japanese Imari porcelain jar",
        "with cobalt-blue underglaze and iron-red and gold overglaze decoration, showing bold floral panels and patterned reserves",
        "a Korean Joseon iron-painted porcelain jar",
        "with freely brushed iron-brown decoration on a warm white porcelain body, showing dragons, bamboo, or abbreviated floral motifs",
    ),
    (
        "a Japanese Imari porcelain plate",
        "with cobalt-blue underglaze and iron-red and gold overglaze decoration, showing radial floral panels and patterned reserves",
        "a Japanese Kutani porcelain plate",
        "with polychrome overglaze enamels in green, yellow, red, purple, and blue, showing bold floral and landscape panels",
    ),
    (
        "a German Meissen porcelain figurine",
        "with a glossy white porcelain body and delicately modeled forms, painted in soft overglaze colors with gilded Rococo details",
        "an Italian Capodimonte porcelain figurine",
        "with highly modeled porcelain drapery and flowers, painted in warm pastel colors with theatrical Baroque gestures",
    ),
    (
        "a French Sèvres porcelain vase",
        "with a saturated bleu-céleste ground, gilded bronze-like ornament, and painted floral or courtly reserve panels",
        "a Russian Imperial Porcelain cobalt-net vase",
        "with a white porcelain body covered by a cobalt-blue lattice, gold intersections, and small radiating rosette motifs",
    ),
    (
        "a Chinese famille-rose porcelain vase",
        "with opaque pink, green, yellow, and turquoise overglaze enamels, showing peonies, birds, and auspicious garden scenes",
        "a Hungarian Herend porcelain vase",
        "with fine polychrome enamel painting and gilded rims, showing stylized butterflies, flowers, and scale-like fishnet patterns",
    ),
    (
        "a Chinese Dehua blanc-de-Chine porcelain figurine",
        "with a smooth ivory-white porcelain surface, softly modeled robes, and restrained carved details without painted color",
        "a Japanese Hirado porcelain figurine",
        "with a fine white porcelain body and delicate underglaze-blue accents, showing crisp miniature modeling and textile details",
    ),
    (
        "a Chinese celadon ceramic bowl",
        "with a translucent blue-green glaze over a restrained stoneware form, showing subtle carved lotus petals beneath the glaze",
        "a Vietnamese Lý–Trần celadon ceramic bowl",
        "with a pale jade-green glaze and lightly crackled surface, showing carved lotus petals and combed floral decoration",
    ),

    # 11-20: glazed earthenware and stoneware
    (
        "a Korean Goryeo celadon maebyeong vase",
        "with a luminous green celadon glaze and pale slip inlay, showing cranes, clouds, and chrysanthemum medallions",
        "a Thai Sawankhalok celadon jar",
        "with an olive-green crackled glaze on a sturdy stoneware body, showing incised lotus petals and simple ring bands",
    ),
    (
        "a Dutch blue-and-white Delftware plate",
        "made of opaque white tin-glazed earthenware, painted in crisp cobalt blue with canal-house, tulip, and ornamental border motifs",
        "a Slovak Modra faience plate",
        "made of pale glazed earthenware, painted in cobalt blue, green, yellow, and manganese with stylized birds and folk flowers",
    ),
    (
        "a Dutch blue-and-white Delftware vase",
        "made of opaque white tin-glazed earthenware, painted in crisp cobalt blue with canal-house, tulip, and ornamental border motifs",
        "a Tunisian Qallaline faience vase",
        "made of tin-glazed earthenware, painted in cobalt blue, green, ochre, and manganese with cypress, arch, and floral motifs",
    ),
    (
        "an Italian Renaissance maiolica plate",
        "made of opaque white tin-glazed earthenware, painted in cobalt blue, ochre yellow, copper green, and manganese purple with a figurative medallion and scrolling foliage",
        "a French Nevers faience plate",
        "made of white tin-glazed earthenware, painted in deep blue, yellow, and manganese with heraldic, pastoral, and symmetrical floral motifs",
    ),
    (
        "an Italian Renaissance maiolica vase",
        "made of opaque white tin-glazed earthenware, painted in cobalt blue, ochre yellow, copper green, and manganese purple with a figurative medallion and scrolling foliage",
        "a Spanish Manises lusterware vase",
        "made of tin-glazed earthenware with coppery metallic luster, showing cobalt-blue vines, heraldic shields, and repeating leaf motifs",
    ),
    (
        "a Mexican Talavera de Puebla plate",
        "made of opaque white tin-glazed earthenware, decorated with cobalt-blue, yellow, green, and orange painted floral and geometric bands",
        "a Romanian Horezu pottery plate",
        "made of glazed earthenware with combed slip decoration, showing brown, green, blue, and ochre spirals, rosettes, and the Horezu rooster",
    ),
    (
        "an Ottoman İznik ceramic plate",
        "with a white fritware body, cobalt blue, turquoise, emerald green, and raised bole-red painting, showing tulips, carnations, and saz leaves",
        "an Armenian Kütahya ceramic plate",
        "with a white glazed body and vivid blue, green, yellow, and red painting, showing small rosettes, crosses, birds, and dense floral borders",
    ),
    (
        "a Moroccan Fes ceramic bowl",
        "with a white glazed body painted in cobalt blue, green, and yellow, showing tightly organized star, arabesque, and palmette motifs",
        "an Algerian Kabyle pottery bowl",
        "made of hand-built earthenware with red, black, and white painted decoration, showing bold triangles, chevrons, and protective geometric signs",
    ),
    (
        "a Japanese Raku tea bowl",
        "with a hand-shaped low-fired body and irregular black glaze, showing a softly warped rim and tactile matte-to-gloss surface",
        "a Japanese Onta ware bowl",
        "made of warm stoneware with slip-trailed and combed decoration, showing rhythmic white, brown, and ochre bands made by traditional wheel techniques",
    ),
    (
        "a Polish Bolesławiec stoneware jug",
        "with a cream glazed body stamped in cobalt blue, showing repeated dots, peacock-eye rosettes, and dense circular folk patterns",
        "a Bulgarian Troyan pottery jug",
        "with a dark glazed earthenware body and feathered slip decoration, showing flowing brown, green, yellow, and cream wave patterns",
    ),

    # 21-30: additional ceramic traditions
    (
        "a Japanese Raku tea bowl",
        "with a hand-shaped low-fired body and irregular black glaze, showing a softly warped rim and tactile matte-to-gloss surface",
        "a Korean buncheong tea bowl",
        "made of gray stoneware brushed with white slip, showing stamped, incised, or freely painted decoration beneath a translucent glaze",
    ),
    (
        "a Chinese sancai glazed ceramic jar",
        "with flowing amber, green, and cream lead glazes over a rounded earthenware body, showing lively drips and molded relief bands",
        "a Peruvian Chulucanas pottery jar",
        "with a smooth burnished clay surface in black, cream, and warm brown, showing negative-painted geometric bands and abstract figures",
    ),
    (
        "an ancient Greek black-figure ceramic amphora",
        "with glossy black silhouettes on warm red clay, showing mythological figures arranged in horizontal narrative registers",
        "an ancient Cypriot bichrome ceramic amphora",
        "with red and black painted lines on a pale clay body, showing concentric circles, birds, and compact geometric panels",
    ),
    (
        "an Acoma Pueblo pottery jar",
        "with a pale clay body and precise black-and-red painting, showing fine-line rain, stepped-cloud, and interlocking geometric motifs",
        "a Catawba burnished pottery jar",
        "made of unglazed hand-coiled clay with a smoky brown surface, showing a highly polished finish and restrained incised details",
    ),
    (
        "a Mexican Mata Ortiz pottery jar",
        "with a thin-walled polished clay body and intricate black-and-red painting, showing interlocking geometric and Mimbres-inspired animal motifs",
        "a Serbian Zlakusa pottery jar",
        "made from clay mixed with ground calcite, showing a warm unglazed surface, rounded cooking-vessel form, and sparse incised bands",
    ),
    (
        "a Moroccan Safi pottery vase",
        "with a glazed earthenware body painted in saturated blue, green, yellow, and brown, showing bold arabesques and floral medallions",
        "a Tunisian Nabeul pottery vase",
        "with a white glazed body painted in cobalt blue, turquoise, yellow, and green, showing fish, flowers, and Mediterranean geometric borders",
    ),
    (
        "an English Wedgwood jasperware vase",
        "with a matte blue stoneware body and crisp white relief decoration, showing classical figures, laurel swags, and geometric borders",
        "a Portuguese Caldas da Rainha ceramic vase",
        "with brightly glazed sculptural earthenware, showing naturalistic leaves, fruit, insects, and textured relief surfaces",
    ),
    (
        "a Chinese Jun ware ceramic bowl",
        "with a thick opalescent blue glaze and irregular purple splashes, showing a simple rounded form and softly pooled color",
        "a Japanese Hagi ware tea bowl",
        "with a warm porous stoneware body and milky white crackled glaze, showing a softly irregular foot and restrained ash-toned surface",
    ),
    (
        "a Japanese Bizen stoneware vase",
        "with an unglazed reddish-brown body marked by natural ash deposits and flame flashes, showing an austere asymmetrical form",
        "a French Puisaye stoneware vase",
        "with a dark salt-glazed stoneware body and earthy brown-to-gray surface, showing restrained incised bands and robust hand-thrown proportions",
    ),
    (
        "a Chinese Longquan celadon vase",
        "with a thick translucent sea-green glaze over a balanced stoneware form, showing carved lotus petals and softly defined ridges",
        "a Thai Si Satchanalai celadon vase",
        "with an olive-green crackled glaze over gray stoneware, showing incised lotus petals, ring bands, and a sturdy high-shouldered profile",
    ),

    # 31-40: lacquer and painted wood
    (
        "a Japanese maki-e lacquerware box",
        "with deep black lacquer sprinkled with gold and silver powder, showing refined cranes, pine branches, and flowing landscape motifs",
        "a Thai lai rod nam lacquerware box",
        "with black lacquer covered by gold-leaf resist decoration, showing dense flame, deity, and scrolling vegetal patterns",
    ),
    (
        "a Japanese maki-e lacquerware tray",
        "with deep black lacquer sprinkled with gold and silver powder, showing refined cranes, pine branches, and flowing landscape motifs",
        "a Burmese shwe zawa lacquerware tray",
        "with a glossy black lacquer ground and engraved gold-leaf decoration, showing courtly figures, mythical animals, and tightly filled borders",
    ),
    (
        "a Chinese carved cinnabar-lacquer box",
        "with thick vermilion lacquer carved in deep relief, showing peonies, dragons, and layered landscape scenes over geometric grounds",
        "a Bhutanese lacquered wooden box",
        "with a warm red or black lacquer surface, showing hand-painted lotus petals, cloud bands, and restrained Buddhist auspicious motifs",
    ),
    (
        "a Japanese raden lacquerware lidded box",
        "with deep black lacquer and iridescent mother-of-pearl shell inlay forming fine bird, flower, and geometric motifs",
        "a Mexican Olinalá lacquerware lidded box",
        "made from linaloe wood, with vivid red, ochre, green, blue, and white floral and animal decoration and fine incised rayado patterns",
    ),
    (
        "a Japanese raden lacquerware tray",
        "with deep black lacquer and iridescent mother-of-pearl shell inlay forming fine bird, flower, and geometric motifs",
        "a Korean najeonchilgi lacquerware tray",
        "with black lacquer and luminous mother-of-pearl inlay, showing dense chrysanthemum scrolls, cranes, and repeated geometric borders",
    ),
    (
        "a Japanese maki-e lacquerware vase",
        "with deep black lacquer sprinkled with gold and silver powder, showing refined cranes, pine branches, and flowing landscape motifs",
        "a Vietnamese sơn mài lacquerware vase",
        "with layered black, red, and gold lacquer polished to depth, showing stylized bamboo, village, and eggshell-inlay motifs",
    ),
    (
        "a Chinese black-lacquer wooden bowl",
        "with a smooth deep-black lacquer surface and restrained red or gold linear decoration, showing a thin balanced profile",
        "a Burmese yun lacquerware bowl",
        "with black lacquer incised and filled with red, green, and yellow pigments, showing zodiac figures and dense geometric bands",
    ),
    (
        "a Russian Khokhloma painted wooden bowl",
        "with a glossy black and gold ground painted in bright red, showing curling berries, leaves, and sweeping floral stems",
        "a Ryukyuan lacquerware bowl",
        "with a glossy vermilion lacquer surface and delicate gold decoration, showing hibiscus, birds, and island landscape motifs",
    ),
    (
        "a Japanese Tsugaru-nuri lacquerware box",
        "with many polished lacquer layers forming mottled red, black, yellow, and green patterns across a smooth geometric form",
        "a Kashmiri papier-mâché lacquer box",
        "with a lacquered paper-pulp body painted in fine gold and jewel colors, showing chinar leaves, flowers, and dense arabesques",
    ),
    (
        "a Chinese lacquered folding screen",
        "with deep black lacquer, gilded outlines, and carved or painted garden scenes showing pavilions, birds, and flowering trees",
        "a Vietnamese sơn mài lacquer screen",
        "with layered black, red, and gold lacquer polished to depth, showing stylized village scenes, bamboo, and luminous eggshell inlay",
    ),

    # 41-50: enamel and metal inlay
    (
        "a Chinese cloisonné vase",
        "with saturated turquoise and cobalt-blue enamel cells separated by fine gilded metal wires, showing lotus, cloud, and auspicious motifs",
        "a Georgian Minankari enamel vase",
        "with translucent jewel-toned enamel over engraved silver, showing compact flowers, birds, and scrolling vines outlined by fine metalwork",
    ),
    (
        "a Chinese cloisonné box",
        "with saturated turquoise and cobalt-blue enamel cells separated by fine gilded metal wires, showing lotus, cloud, and auspicious motifs",
        "a Korean chilbo enamel box",
        "with luminous colored enamel over a metal base, showing simplified plum blossoms, cranes, and geometric motifs in compact panels",
    ),
    (
        "a Russian Fabergé-style guilloché enamel box",
        "with translucent enamel over machine-engraved radiating patterns, showing pastel color fields, gold mounts, and jeweled accents",
        "an Indian Jaipur meenakari enamel box",
        "with opaque red, green, blue, and white enamel on gilded metal, showing peacocks, flowers, and dense scrolling foliage",
    ),
    (
        "a Japanese shippō cloisonné enamel box",
        "with smooth colored enamel cells divided by fine silver wires, showing chrysanthemums, butterflies, and softly graded pictorial grounds",
        "a Russian Rostov finift painted-enamel box",
        "with miniature enamel painting on a white metal-mounted plaque, showing delicate flowers, saints, or pastoral scenes in pastel colors",
    ),
    (
        "a Spanish Toledo damascened box",
        "made of blackened steel inlaid with gold and silver, showing dense Moorish arabesques and geometric filigree patterns",
        "a Japanese nunome-zōgan metal-inlay box",
        "made of dark iron overlaid with fine gold and silver textile-like inlay, showing restrained waves, flowers, and geometric bands",
    ),
    (
        "a Spanish Toledo damascened vase",
        "made of blackened steel inlaid with gold and silver, showing dense Moorish arabesques and geometric filigree patterns",
        "a Korean ipsa metal-inlay vase",
        "made of dark patinated metal inlaid with fine silver wire, showing cranes, clouds, and sparse linear landscape motifs",
    ),
    (
        "an Indian Bidriware vase",
        "made of blackened zinc alloy inlaid with bright silver, showing poppy flowers, geometric lattices, and flowing arabesques",
        "a Syrian Damascene inlaid-brass vase",
        "made of darkened brass inlaid with silver and copper, showing calligraphic bands, interlaced stars, and scrolling arabesques",
    ),
    (
        "an Indian Bidriware box",
        "made of blackened zinc alloy inlaid with bright silver, showing poppy flowers, geometric lattices, and flowing arabesques",
        "an Egyptian Mamluk-style inlaid-brass box",
        "made of brass inlaid with silver and copper, showing monumental calligraphy, interlaced medallions, and geometric borders",
    ),
    (
        "a Chinese cloisonné enamel plate",
        "with saturated turquoise and cobalt-blue enamel cells separated by fine gilded metal wires, showing lotus, cloud, and auspicious motifs",
        "a French Limoges painted-enamel plate",
        "with opaque and translucent enamel painted over copper, showing Renaissance figures, grisaille shading, and jewel-toned borders",
    ),
    (
        "a Persian minakari enamel plate",
        "with bright blue, turquoise, red, and white enamel over copper, showing dense floral arabesques and central medallions",
        "a Georgian Minankari enamel plate",
        "with translucent jewel-toned enamel over engraved silver, showing compact flowers, birds, and scrolling vines outlined by fine metalwork",
    ),

    # 51-60: engraved, cast, and filigree metalwork
    (
        "a Moroccan engraved-brass tray",
        "with a warm polished brass surface densely chased with radiating stars, arabesques, and scalloped geometric borders",
        "a Persian qalamzani copper tray",
        "with a darkened chased copper surface, showing fine hunting scenes, birds, flowers, and concentric arabesque bands",
    ),
    (
        "an Ottoman tinned-copper ewer",
        "with a reflective silver-toned tinned surface, tall curved spout, and chased tulip, cypress, and arabesque decoration",
        "a Bosnian engraved-copper coffee ewer",
        "with a warm copper body, long curved spout, and hand-hammered rosette, star, and interlaced geometric motifs",
    ),
    (
        "a Persian silver-filigree box",
        "with delicate silver wire scrollwork forming an airy surface, showing repeated palmettes, rosettes, and geometric borders",
        "a Yemeni silver-filigree box",
        "with dense twisted silver wire, granulation, and small applied bosses, showing compact geometric and amuletic patterns",
    ),
    (
        "a Mexican Taxco silver bowl",
        "with a polished hand-hammered silver surface and bold modern curves, showing restrained geometric or pre-Columbian-inspired accents",
        "an Indonesian Kotagede silver bowl",
        "with a softly oxidized silver surface worked in repoussé and filigree, showing lotus scrolls, vines, and fine stippled grounds",
    ),
    (
        "an Indian Dhokra bronze figurine",
        "made by lost-wax casting with an earthy bronze surface, showing elongated limbs, coiled-wire textures, and stylized tribal ornament",
        "a Ghanaian Asante brass goldweight figurine",
        "made by lost-wax casting in compact geometric form, showing a proverb-inspired human, animal, or tool motif with textured details",
    ),
    (
        "a Benin bronze relief plaque",
        "made of cast brass with high-relief court figures, showing patterned robes, coral regalia, and a dense hierarchical composition",
        "a Nepalese repoussé copper plaque",
        "made of hammered copper worked from the reverse, showing a Buddhist deity, lotus throne, and flame-like halo with gilded details",
    ),
    (
        "a Japanese cast-iron tetsubin teapot",
        "with a dark textured iron body, arched handle, and restrained relief decoration showing hailstone, pine, or geometric motifs",
        "an Azerbaijani engraved-copper teapot",
        "with a warm hammered copper body, curved spout, and incised floral medallions, boteh motifs, and geometric borders",
    ),
    (
        "a Russian brass samovar",
        "with a polished urn-shaped body, side handles, and ornate spigot, showing engraved floral bands and architectural details",
        "an Iranian Qajar brass samovar",
        "with a tall faceted brass body, curved handles, and engraved panels showing birds, flowers, and courtly motifs",
    ),
    (
        "a Tibetan silver ritual box",
        "with repoussé silver, turquoise settings, and gilded details, showing lotus petals, cloud scrolls, and Buddhist auspicious symbols",
        "a Balkan silver-filigree reliquary box",
        "with dense twisted silver wire and small granulated bosses, showing crosses, rosettes, and compact geometric latticework",
    ),
    (
        "a Chinese bronze ritual bowl",
        "with a dark patinated bronze surface and cast relief, showing taotie-like masks, leiwen spirals, and strong symmetrical handles",
        "a Korean bangjja bronze bowl",
        "made of hand-hammered bronze with a warm golden surface, showing a clean resonant form, subtle hammer marks, and minimal ornament",
    ),

    # 61-70: silk, brocade, and ikat
    (
        "a Japanese silk kimono",
        "with a lustrous woven ground and dyed or embroidered decoration, showing cranes, cherry blossoms, flowing water, and seasonal motifs",
        "a Lao sinh silk skirt",
        "with supplementary-weft silk bands and intricate patterned borders, showing diamonds, serpents, temple forms, and saturated jewel colors",
    ),
    (
        "a Japanese silk kimono",
        "with a lustrous woven ground and dyed or embroidered decoration, showing cranes, cherry blossoms, flowing water, and seasonal motifs",
        "a Cambodian hol silk robe",
        "made from weft-ikat silk with softly feathered pattern edges, showing repeating lozenges, flowers, and temple-inspired motifs",
    ),
    (
        "an Indian Banarasi brocade sari",
        "woven in rich silk with gold zari, showing dense floral buta, scrolling vines, and elaborate architectural borders",
        "a Bhutanese kushuthara silk kira",
        "woven in bright silk with intricate supplementary-weft patterning, showing small geometric flowers, diamonds, and horizontal bands",
    ),
    (
        "an Indian Banarasi brocade textile panel",
        "woven in rich silk with gold zari, showing dense floral buta, scrolling vines, and elaborate architectural borders",
        "a Turkmen keteni silk textile panel",
        "woven in narrow bands of luminous red, yellow, green, and purple silk, showing crisp stripes and restrained geometric accents",
    ),
    (
        "a Chinese silk-embroidery panel",
        "worked in fine multicolored silk thread, showing peonies, birds, butterflies, and subtle long-and-short stitch shading",
        "a Vietnamese silk-embroidery panel",
        "worked in fine colored silk thread with delicate tonal blending, showing village landscapes, lotus ponds, and graceful birds",
    ),
    (
        "a Chinese silk-brocade robe",
        "woven with colored silk and gold thread, showing dragons, clouds, waves, and tightly organized auspicious emblems",
        "an Assamese muga-silk robe",
        "woven from naturally golden silk with red, green, and black supplementary motifs, showing stylized flowers and geometric borders",
    ),
    (
        "a Japanese brocade obi sash",
        "woven in silk and metallic thread, showing large chrysanthemums, cranes, fans, and rhythmic geometric grounds",
        "a Korean gold-thread brocade sash",
        "woven in colored silk with metallic thread, showing compact clouds, peonies, phoenixes, and repeating geometric borders",
    ),
    (
        "an Indonesian ikat textile panel",
        "woven from resist-dyed threads with softly feathered edges, showing bold diamonds, ancestor figures, and rhythmic color bands",
        "an Iban pua kumbu textile panel",
        "woven in warp ikat with deep red, black, and cream yarns, showing powerful hooked figures, spirit motifs, and mirrored geometric bands",
    ),
    (
        "an Indonesian ikat textile panel",
        "woven from resist-dyed threads with softly feathered edges, showing bold diamonds, ancestor figures, and rhythmic color bands",
        "a Sumbanese hinggi textile panel",
        "woven in warp ikat with indigo, rust red, and cream, showing horses, roosters, skull trees, and large heraldic figures",
    ),
    (
        "an Indian Patola double-ikat textile panel",
        "woven from precisely resist-dyed silk warp and weft, showing crisp dancing figures, elephants, flowers, and geometric grids",
        "a Timorese tais textile panel",
        "woven in cotton warp ikat with red, black, yellow, and white bands, showing hooks, diamonds, and locally specific ancestral motifs",
    ),

    # 71-80: dyed, printed, embroidered, and patchwork textiles
    (
        "an Indonesian batik cloth",
        "with wax-resist dyeing in indigo, brown, and cream, showing parang diagonals, kawung rosettes, and densely repeated floral motifs",
        "a Nigerian adire cloth",
        "with indigo resist-dyed cotton, showing bold white circles, grids, comb-like marks, and hand-drawn or stitched geometric patterns",
    ),
    (
        "an Indonesian batik textile panel",
        "with wax-resist dyeing in indigo, brown, and cream, showing parang diagonals, kawung rosettes, and densely repeated floral motifs",
        "a Japanese katazome textile panel",
        "with stencil-resist dyeing on cotton in indigo and white, showing crisp waves, chrysanthemums, birds, and repeating geometric motifs",
    ),
    (
        "an Indian Ajrakh block-printed cloth",
        "with layered resist printing in indigo, madder red, black, and white, showing precisely repeated stars and interlocking geometry",
        "a Chinese Nantong blue-calico cloth",
        "with indigo resist dyeing on cotton, showing bold white flowers, birds, fish, and auspicious folk motifs against a deep-blue ground",
    ),
    (
        "a Mexican Otomi embroidery panel",
        "worked in bright flat satin stitch on pale cotton, showing mirrored birds, deer, flowers, and fantastical animals",
        "a Palestinian tatreez embroidery panel",
        "worked in dense red and multicolored cross-stitch, showing cypress trees, stars, feathers, and region-specific geometric motifs",
    ),
    (
        "a Mexican Otomi embroidered blouse",
        "worked in bright flat satin stitch, showing mirrored birds, deer, flowers, and fantastical animals around the neckline and sleeves",
        "a Guatemalan huipil blouse",
        "woven and embroidered in saturated colors, showing diamonds, birds, maize, and community-specific geometric bands",
    ),
    (
        "a Japanese sashiko textile panel",
        "stitched with white running thread on indigo cotton, showing repeating hemp-leaf, wave, and linked-diamond reinforcement patterns",
        "a Romanian ie-embroidery textile panel",
        "worked in red, black, blue, and metallic thread, showing vertical floral bands, diamonds, and region-specific geometric motifs",
    ),
    (
        "a Hungarian Matyó embroidery panel",
        "worked in saturated red, pink, blue, yellow, and green thread, showing densely packed roses and curling floral sprays",
        "a Slovak Detva embroidery panel",
        "worked in bright chain stitch and satin stitch, showing compact geometric flowers, hearts, and rhythmic folk borders",
    ),
    (
        "an Indian kantha quilt",
        "made from layered cotton joined by dense running stitches, showing rippling lines, lotus flowers, animals, and narrative folk scenes",
        "a Korean bojagi patchwork wrapping cloth",
        "assembled from translucent or opaque fabric rectangles, showing asymmetrical geometric seams and restrained jewel-toned color blocks",
    ),
    (
        "a Panamanian mola textile panel",
        "made with layered reverse appliqué in vivid colors, showing animals, plants, and maze-like geometric contours",
        "a Ukrainian Hutsul embroidery panel",
        "worked in dense cross-stitch with red, orange, black, and green thread, showing diamonds, hooks, and tightly repeated geometric bands",
    ),
    (
        "a Ghanaian kente cloth",
        "woven in narrow strips of bright silk or rayon, showing alternating geometric blocks in gold, green, red, blue, and black",
        "a Bhutanese yathra wool textile",
        "woven in thick colored wool with supplementary-weft motifs, showing repeating diamonds, flowers, and horizontal geometric bands",
    ),

    # 81-90: carpets, rugs, kilims, and felt
    (
        "a Persian floral carpet",
        "with a dense wool pile and central medallion, showing curving vines, palmettes, and flowers in deep red, blue, and ivory",
        "a Turkmen Tekke carpet",
        "with a deep red wool pile and repeated octagonal guls, showing compact black, white, and dark-blue tribal geometry",
    ),
    (
        "a Persian floral carpet",
        "with a dense wool pile and central medallion, showing curving vines, palmettes, and flowers in deep red, blue, and ivory",
        "an Azerbaijani Kuba carpet",
        "with a tightly knotted wool pile, showing angular medallions, stylized animals, and crisp blue, red, and ivory geometric borders",
    ),
    (
        "a Persian floral carpet",
        "with a dense wool pile and central medallion, showing curving vines, palmettes, and flowers in deep red, blue, and ivory",
        "an Armenian Karabakh carpet",
        "with a wool pile in saturated red, blue, green, and cream, showing bold floral medallions and elongated geometric forms",
    ),
    (
        "a Moroccan Beni Ourain wool rug",
        "with a thick cream wool pile and sparse dark-brown lines, showing large asymmetrical diamonds and open geometric spacing",
        "a Ukrainian Hutsul lizhnyk wool rug",
        "with a shaggy woven wool surface in gray, cream, brown, and muted color, showing stepped diamonds and rhythmic zigzag bands",
    ),
    (
        "a Turkish flat-woven kilim",
        "with tightly interlocked wool wefts in red, blue, orange, and cream, showing hooked diamonds, ram horns, and stepped medallions",
        "a Romanian Oltenian kilim",
        "with a flat-woven wool surface in warm red, green, blue, and cream, showing stylized flowers, birds, and vertical bouquet arrangements",
    ),
    (
        "a Turkish flat-woven kilim",
        "with tightly interlocked wool wefts in red, blue, orange, and cream, showing hooked diamonds, ram horns, and stepped medallions",
        "a Serbian Pirot kilim",
        "with a reversible flat weave and crisp red, black, blue, and white geometry, showing turtles, hooks, and repeated protective motifs",
    ),
    (
        "a Scottish tartan wool blanket",
        "woven in colored wool with intersecting horizontal and vertical stripes, forming a balanced repeating clan-associated check pattern",
        "a Swedish rya wool rug",
        "with a long wool pile and woven backing, showing softly geometric diamonds, stripes, and abstract color fields",
    ),
    (
        "a Navajo wool rug",
        "with a tightly woven flat surface and strong red, black, gray, and cream geometry, showing stepped diamonds and serrated bands",
        "a Kyrgyz shyrdak felt rug",
        "made from cut and inlaid wool felt in contrasting colors, showing mirrored horn motifs, scrolling forms, and bold outlined geometry",
    ),
    (
        "a Persian floral carpet",
        "with a dense wool pile and central medallion, showing curving vines, palmettes, and flowers in deep red, blue, and ivory",
        "a Tibetan khaden carpet",
        "with a compact wool pile in red, blue, yellow, and cream, showing snow lions, lotus flowers, clouds, or checkerboard motifs",
    ),
    (
        "a Moroccan geometric wool rug",
        "with a hand-knotted or flat-woven wool surface, showing bold diamonds, zigzags, and irregular bands in warm earth colors",
        "an Afghan Baluch wool rug",
        "with a dark wool pile in deep red, brown, navy, and ivory, showing repeated prayer niches, hooked medallions, and compact tribal borders",
    ),

    # 91-100: glass, basketry, leather, wood, and stone
    (
        "a Venetian Murano glass vase",
        "made from vividly colored blown glass with layered canes, showing flowing curves, millefiori accents, and polished translucent depth",
        "a Palestinian Hebron blown-glass vase",
        "made from recycled glass in translucent turquoise and green, showing a softly irregular hand-blown body, bubbles, and applied glass trails",
    ),
    (
        "a Bohemian cut-glass vase",
        "made from clear or ruby-overlaid crystal with deep wheel-cut facets, showing stars, fans, and sharply sparkling geometric panels",
        "a Czech Železný Brod art-glass vase",
        "made from hand-shaped colored glass with clean modern curves, showing layered transparent tones, internal bubbles, and restrained sculptural forms",
    ),
    (
        "a Venetian millefiori glass-bead necklace",
        "made from multicolored glass canes sliced into flower-like beads, showing dense red, blue, yellow, white, and green mosaic patterns",
        "a Ghanaian Krobo powder-glass-bead necklace",
        "made from molded recycled glass powder, showing opaque hand-painted stripes, dots, chevrons, and saturated earth and jewel colors",
    ),
    (
        "a Japanese woven-bamboo basket",
        "made from split bamboo in a refined open weave, showing precise hexagonal plaiting, curved handles, and an asymmetrical sculptural profile",
        "a Philippine nito-vine basket",
        "woven from dark nito vine over a lighter structural frame, showing tight black-and-tan spirals, diamonds, and rhythmic geometric bands",
    ),
    (
        "a Navajo coiled basket",
        "woven from plant fibers in a shallow circular form, showing a central opening and radiating red, black, and natural spiral motifs",
        "a Panamanian Wounaan chunga basket",
        "coiled from fine palm fiber, showing exceptionally tight weaving and vivid animal, plant, or geometric motifs in black and bright colors",
    ),
    (
        "a Moroccan tooled-leather box",
        "covered in warm brown leather with stamped and painted ornament, showing stars, arabesques, and scalloped geometric borders",
        "a Tuareg tooled-leather box",
        "covered in dark leather with incised and colored decoration, showing triangles, crosses, zigzags, and applied metal studs",
    ),
    (
        "an Italian tooled-leather wall panel",
        "made from embossed and gilded leather, showing scrolling acanthus leaves, heraldic forms, and warm brown-and-gold relief",
        "a Spanish Córdoba guadamecí leather panel",
        "made from silvered, painted, and embossed leather, showing Renaissance arabesques, flowers, and richly colored metallic relief",
    ),
    (
        "a Russian matryoshka wooden doll",
        "with a turned wooden body painted in bright colors, showing a smiling figure in a floral headscarf and apron",
        "a Japanese Tsugaru kokeshi wooden doll",
        "with a lathe-turned cylindrical body and rounded head, showing restrained painted rings, chrysanthemum motifs, and a simple face",
    ),
    (
        "a Chinese carved-jade pendant",
        "made from polished pale green or white jade, showing pierced clouds, dragons, or floral scrolls in smooth low relief",
        "a Māori pounamu hei tiki pendant",
        "carved from translucent greenstone in a compact ancestral figure, showing an inclined head, large eyes, and curved limbs",
    ),
    (
        "an Inuit soapstone figurine",
        "carved from dark gray-green stone with rounded tactile surfaces, showing a compact animal or human figure with restrained incised details",
        "a Kenyan Kisii soapstone figurine",
        "carved from pale soapstone and polished smooth, showing an abstract intertwined human or animal form with softly rounded contours",
    ),
]


def compact_suffix(text: str) -> str:
    """Remove low-information modifiers without cutting semantic phrases."""
    text = re.sub(r"\s+", " ", text.strip())
    low_information_modifiers = (
        "refined", "dense", "bold", "glossy", "delicately", "soft", "highly",
        "warm", "saturated", "fine", "smooth", "softly", "restrained", "crisp",
        "subtle", "luminous", "sturdy", "vivid", "small", "tightly", "traditional",
        "lively", "precise", "intricate", "flowing", "bright", "strong",
    )
    for modifier in low_information_modifiers:
        text = re.sub(rf"\b{re.escape(modifier)}\b\s*", "", text, flags=re.IGNORECASE)
    phrase_replacements = (
        (r"\s+beneath a transparent glaze", ""),
        (r"with a white porcelain body covered by a ", "with "),
        (r"with an? ivory-white porcelain surface", "with an ivory-white surface"),
        (r"with a white porcelain body and ", "with "),
        (r"with a translucent ([^,]+) glaze over a stoneware form", r"with translucent \1 glaze"),
        (r"made of opaque white tin-glazed earthenware,\s*", "with opaque white tin glaze, "),
        (r"made of white tin-glazed earthenware,\s*", "with white tin glaze, "),
        (r"made of tin-glazed earthenware,\s*", "with tin glaze, "),
        (r"on a pale glazed body", "on pale glaze"),
        (r"on a white porcelain body", "on white porcelain"),
    )
    for pattern, replacement in phrase_replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    return text.strip(" ,;")


def build_record(a_short: str, a_suffix: str, b_short: str, b_suffix: str) -> dict:
    a_suffix = compact_suffix(a_suffix)
    b_suffix = compact_suffix(b_suffix)
    a_long = f"{a_short} {a_suffix}"
    b_long = f"{b_short} {b_suffix}"
    return {
        "文化物体A": a_short,
        "文化物体B": b_short,
        "文化物体A的长文本描述": a_long,
        "文化物体B的长文本描述": b_long,
        "组合Prompt SS": SHELL.format(A=a_short, B=b_short),
        "组合Prompt SL": SHELL.format(A=a_short, B=b_long),
        "组合Prompt LL": SHELL.format(A=a_long, B=b_long),
    }


def validate(records: list[dict]) -> None:
    expected_keys = [
        "文化物体A",
        "文化物体B",
        "文化物体A的长文本描述",
        "文化物体B的长文本描述",
        "组合Prompt SS",
        "组合Prompt SL",
        "组合Prompt LL",
    ]
    assert len(records) == 100, len(records)
    assert len({(r["文化物体A"], r["文化物体B"]) for r in records}) == 100
    for i, record in enumerate(records, 1):
        assert list(record) == expected_keys, (i, list(record))
        assert record["文化物体A的长文本描述"].startswith(record["文化物体A"] + " "), i
        assert record["文化物体B的长文本描述"].startswith(record["文化物体B"] + " "), i
        assert record["组合Prompt SS"] == SHELL.format(
            A=record["文化物体A"], B=record["文化物体B"]
        ), i
        assert record["组合Prompt SL"] == SHELL.format(
            A=record["文化物体A"], B=record["文化物体B的长文本描述"]
        ), i
        assert record["组合Prompt LL"] == SHELL.format(
            A=record["文化物体A的长文本描述"], B=record["文化物体B的长文本描述"]
        ), i


def main() -> None:
    records = [build_record(*pair) for pair in PAIRS]
    validate(records)
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "pairs": len(records), "validation": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
