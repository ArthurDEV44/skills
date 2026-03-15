# Structured Data Reference — JSON-LD Templates (2026)

All templates use JSON-LD format as recommended by Google. Place inside
`<script type="application/ld+json">` in `<head>` or `<body>`.

## Priority Tier System

| Tier | Schema Type | Rich Result | Impact |
|------|------------|-------------|--------|
| P0 | Organization | Knowledge panel | Brand authority |
| P0 | WebSite + SearchAction | Sitelinks search box | Navigation |
| P0 | BreadcrumbList | Breadcrumb trail | All inner pages |
| P1 | Article / BlogPosting | Discover, AI citation | Content sites |
| P1 | FAQPage | FAQ rich results, AI extraction | 20-30% CTR lift |
| P1 | Product + Offer + Review | Shopping rich results | 18-25% CTR lift |
| P1 | LocalBusiness | Local pack, knowledge panel | Local businesses |
| P2 | HowTo | Step-by-step results | Tutorial content |
| P2 | VideoObject | Video carousel | Video content |
| P2 | SoftwareApplication | App panels | Software/SaaS |
| P2 | Review / AggregateRating | Star ratings | Review content |
| P3 | Event | Event listing | Events |
| P3 | Course | Course rich results | Education |
| P3 | Recipe | Recipe carousel | Food content |
| P3 | JobPosting | Job listing | Hiring |

## P0 Templates

### Organization

```json
{
  "@context": "https://schema.org",
  "@type": "Organization",
  "name": "{{company_name}}",
  "url": "https://www.{{domain}}",
  "logo": {
    "@type": "ImageObject",
    "url": "https://www.{{domain}}/logo.png",
    "width": 600,
    "height": 60
  },
  "description": "{{company_description}}",
  "foundingDate": "{{YYYY}}",
  "founder": {
    "@type": "Person",
    "name": "{{founder_name}}"
  },
  "sameAs": [
    "https://twitter.com/{{handle}}",
    "https://linkedin.com/company/{{handle}}",
    "https://github.com/{{handle}}",
    "https://en.wikipedia.org/wiki/{{Company_Name}}"
  ],
  "contactPoint": {
    "@type": "ContactPoint",
    "telephone": "+1-{{phone}}",
    "contactType": "customer service",
    "availableLanguage": ["English", "French"]
  }
}
```

### WebSite + SearchAction (Sitelinks Search Box)

```json
{
  "@context": "https://schema.org",
  "@type": "WebSite",
  "name": "{{site_name}}",
  "url": "https://www.{{domain}}",
  "potentialAction": {
    "@type": "SearchAction",
    "target": {
      "@type": "EntryPoint",
      "urlTemplate": "https://www.{{domain}}/search?q={search_term_string}"
    },
    "query-input": "required name=search_term_string"
  }
}
```

### BreadcrumbList

```json
{
  "@context": "https://schema.org",
  "@type": "BreadcrumbList",
  "itemListElement": [
    {
      "@type": "ListItem",
      "position": 1,
      "name": "Home",
      "item": "https://www.{{domain}}"
    },
    {
      "@type": "ListItem",
      "position": 2,
      "name": "{{category}}",
      "item": "https://www.{{domain}}/{{category_slug}}"
    },
    {
      "@type": "ListItem",
      "position": 3,
      "name": "{{page_title}}",
      "item": "https://www.{{domain}}/{{category_slug}}/{{page_slug}}"
    }
  ]
}
```

## P1 Templates

### Article / BlogPosting

```json
{
  "@context": "https://schema.org",
  "@type": "Article",
  "headline": "{{article_title}}",
  "description": "{{article_summary_155_chars}}",
  "image": {
    "@type": "ImageObject",
    "url": "https://www.{{domain}}/images/{{image_file}}",
    "width": 1200,
    "height": 630
  },
  "datePublished": "{{ISO_8601_published}}",
  "dateModified": "{{ISO_8601_modified}}",
  "author": {
    "@type": "Person",
    "name": "{{author_name}}",
    "url": "https://www.{{domain}}/about/{{author_slug}}",
    "jobTitle": "{{author_title}}",
    "sameAs": [
      "https://twitter.com/{{author_handle}}",
      "https://linkedin.com/in/{{author_handle}}"
    ]
  },
  "publisher": {
    "@type": "Organization",
    "name": "{{site_name}}",
    "logo": {
      "@type": "ImageObject",
      "url": "https://www.{{domain}}/logo.png"
    }
  },
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": "https://www.{{domain}}/blog/{{article_slug}}"
  },
  "wordCount": {{word_count}},
  "articleSection": "{{category}}",
  "keywords": ["{{keyword1}}", "{{keyword2}}", "{{keyword3}}"]
}
```

### FAQPage

```json
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "{{question_1}}",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "{{answer_1}}"
      }
    },
    {
      "@type": "Question",
      "name": "{{question_2}}",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "{{answer_2}}"
      }
    }
  ]
}
```

**FAQPage rules:**
- Questions must be visible on the page (not hidden in accordions for schema purposes)
- Answers should be 1-3 sentences for maximum AI extractability
- Maximum 10 FAQ entries per page recommended
- Each question should match a real user query (use People Also Ask data)

### Product + Offer + AggregateRating

```json
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "{{product_name}}",
  "description": "{{product_description}}",
  "image": [
    "https://www.{{domain}}/products/{{image1}}.jpg",
    "https://www.{{domain}}/products/{{image2}}.jpg"
  ],
  "sku": "{{sku}}",
  "gtin13": "{{ean_barcode}}",
  "mpn": "{{manufacturer_part_number}}",
  "brand": {
    "@type": "Brand",
    "name": "{{brand_name}}"
  },
  "offers": {
    "@type": "Offer",
    "url": "https://www.{{domain}}/products/{{product_slug}}",
    "priceCurrency": "{{currency_code}}",
    "price": "{{price}}",
    "priceValidUntil": "{{YYYY-MM-DD}}",
    "itemCondition": "https://schema.org/NewCondition",
    "availability": "https://schema.org/InStock",
    "seller": {
      "@type": "Organization",
      "name": "{{seller_name}}"
    },
    "shippingDetails": {
      "@type": "OfferShippingDetails",
      "shippingRate": {
        "@type": "MonetaryAmount",
        "value": "{{shipping_cost}}",
        "currency": "{{currency_code}}"
      },
      "deliveryTime": {
        "@type": "ShippingDeliveryTime",
        "handlingTime": {
          "@type": "QuantitativeValue",
          "minValue": 0,
          "maxValue": 1,
          "unitCode": "DAY"
        },
        "transitTime": {
          "@type": "QuantitativeValue",
          "minValue": 1,
          "maxValue": 5,
          "unitCode": "DAY"
        }
      },
      "shippingDestination": {
        "@type": "DefinedRegion",
        "addressCountry": "{{country_code}}"
      }
    },
    "hasMerchantReturnPolicy": {
      "@type": "MerchantReturnPolicy",
      "applicableCountry": "{{country_code}}",
      "returnPolicyCategory": "https://schema.org/MerchantReturnFiniteReturnWindow",
      "merchantReturnDays": 30,
      "returnMethod": "https://schema.org/ReturnByMail",
      "returnFees": "https://schema.org/FreeReturn"
    }
  },
  "aggregateRating": {
    "@type": "AggregateRating",
    "ratingValue": "{{avg_rating}}",
    "bestRating": "5",
    "ratingCount": "{{total_ratings}}",
    "reviewCount": "{{total_reviews}}"
  },
  "review": [
    {
      "@type": "Review",
      "author": {
        "@type": "Person",
        "name": "{{reviewer_name}}"
      },
      "datePublished": "{{ISO_8601}}",
      "reviewRating": {
        "@type": "Rating",
        "ratingValue": "{{rating}}",
        "bestRating": "5"
      },
      "reviewBody": "{{review_text}}"
    }
  ]
}
```

### LocalBusiness

```json
{
  "@context": "https://schema.org",
  "@type": "{{specific_type}}",
  "name": "{{business_name}}",
  "image": "https://www.{{domain}}/storefront.jpg",
  "url": "https://www.{{domain}}",
  "telephone": "+{{phone}}",
  "email": "{{email}}",
  "address": {
    "@type": "PostalAddress",
    "streetAddress": "{{street}}",
    "addressLocality": "{{city}}",
    "addressRegion": "{{state}}",
    "postalCode": "{{zip}}",
    "addressCountry": "{{country_code}}"
  },
  "geo": {
    "@type": "GeoCoordinates",
    "latitude": {{lat}},
    "longitude": {{lng}}
  },
  "openingHoursSpecification": [
    {
      "@type": "OpeningHoursSpecification",
      "dayOfWeek": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
      "opens": "09:00",
      "closes": "18:00"
    },
    {
      "@type": "OpeningHoursSpecification",
      "dayOfWeek": "Saturday",
      "opens": "10:00",
      "closes": "16:00"
    }
  ],
  "priceRange": "{{price_range}}",
  "sameAs": [
    "https://www.google.com/maps/place/{{gmaps_url}}",
    "https://www.yelp.com/biz/{{yelp_slug}}",
    "https://www.facebook.com/{{fb_page}}"
  ],
  "aggregateRating": {
    "@type": "AggregateRating",
    "ratingValue": "{{rating}}",
    "reviewCount": "{{review_count}}"
  }
}
```

**Specific types** for `{{specific_type}}`:
`Restaurant`, `Dentist`, `LegalService`, `AccountingService`, `AutoRepair`,
`BarberShop`, `BeautySalon`, `Brewery`, `CafeOrCoffeeShop`, `Electrician`,
`FinancialService`, `FurnitureStore`, `GasStation`, `GroceryStore`,
`HairSalon`, `HardwareStore`, `HealthClub`, `HomeGoodsStore`, `Hotel`,
`InsuranceAgency`, `Locksmith`, `MedicalClinic`, `Optician`, `PetStore`,
`Pharmacy`, `Plumber`, `RealEstateAgent`, `SportsActivityLocation`,
`Store`, `TravelAgency`, `VeterinaryCare`

## P2 Templates

### HowTo

```json
{
  "@context": "https://schema.org",
  "@type": "HowTo",
  "name": "{{how_to_title}}",
  "description": "{{how_to_summary}}",
  "image": "https://www.{{domain}}/images/{{image}}",
  "totalTime": "PT{{minutes}}M",
  "estimatedCost": {
    "@type": "MonetaryAmount",
    "currency": "{{currency}}",
    "value": "{{cost}}"
  },
  "supply": [
    { "@type": "HowToSupply", "name": "{{supply_1}}" },
    { "@type": "HowToSupply", "name": "{{supply_2}}" }
  ],
  "tool": [
    { "@type": "HowToTool", "name": "{{tool_1}}" }
  ],
  "step": [
    {
      "@type": "HowToStep",
      "name": "{{step_1_title}}",
      "text": "{{step_1_description}}",
      "image": "https://www.{{domain}}/images/step1.jpg",
      "url": "https://www.{{domain}}/how-to/{{slug}}#step1"
    },
    {
      "@type": "HowToStep",
      "name": "{{step_2_title}}",
      "text": "{{step_2_description}}",
      "image": "https://www.{{domain}}/images/step2.jpg",
      "url": "https://www.{{domain}}/how-to/{{slug}}#step2"
    }
  ]
}
```

### VideoObject

```json
{
  "@context": "https://schema.org",
  "@type": "VideoObject",
  "name": "{{video_title}}",
  "description": "{{video_description}}",
  "thumbnailUrl": "https://www.{{domain}}/thumbnails/{{video_id}}.jpg",
  "uploadDate": "{{ISO_8601}}",
  "duration": "PT{{minutes}}M{{seconds}}S",
  "contentUrl": "https://www.{{domain}}/videos/{{video_file}}",
  "embedUrl": "https://www.youtube.com/embed/{{youtube_id}}",
  "interactionStatistic": {
    "@type": "InteractionCounter",
    "interactionType": { "@type": "WatchAction" },
    "userInteractionCount": {{view_count}}
  }
}
```

### SoftwareApplication

```json
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "{{app_name}}",
  "description": "{{app_description}}",
  "applicationCategory": "{{category}}",
  "operatingSystem": "{{os}}",
  "offers": {
    "@type": "Offer",
    "price": "{{price}}",
    "priceCurrency": "{{currency}}"
  },
  "aggregateRating": {
    "@type": "AggregateRating",
    "ratingValue": "{{rating}}",
    "ratingCount": "{{count}}"
  },
  "screenshot": "https://www.{{domain}}/screenshots/{{app}}.png"
}
```

## Validation

Always validate generated JSON-LD:
1. **Syntax:** Valid JSON (use `JSON.parse()` to check)
2. **Schema.org compliance:** Check against https://schema.org types
3. **Google Rich Results Test:** https://search.google.com/test/rich-results
4. **Required fields:** Each type has mandatory fields — missing them = no rich result

## Multiple Schemas Per Page

Use `@graph` to combine multiple schemas on a single page:

```json
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "Organization",
      "name": "..."
    },
    {
      "@type": "BreadcrumbList",
      "itemListElement": [...]
    },
    {
      "@type": "Article",
      "headline": "..."
    }
  ]
}
```

## Next.js App Router Implementation

```typescript
// components/JsonLd.tsx
export function JsonLd({ data }: { data: Record<string, unknown> }) {
  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(data) }}
    />
  );
}

// app/blog/[slug]/page.tsx
import { JsonLd } from '@/components/JsonLd';

export default async function BlogPost({ params }) {
  const post = await getPost(params.slug);

  const articleSchema = {
    '@context': 'https://schema.org',
    '@type': 'Article',
    headline: post.title,
    datePublished: post.publishedAt,
    dateModified: post.updatedAt,
    author: {
      '@type': 'Person',
      name: post.author.name,
      url: `https://example.com/about/${post.author.slug}`,
    },
    publisher: {
      '@type': 'Organization',
      name: 'Site Name',
      logo: { '@type': 'ImageObject', url: 'https://example.com/logo.png' },
    },
    image: post.heroImage,
    description: post.excerpt,
  };

  return (
    <>
      <JsonLd data={articleSchema} />
      <article>{/* content */}</article>
    </>
  );
}
```
